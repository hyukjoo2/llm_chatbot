import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import psycopg2  # PostgreSQL 전용
import psycopg2.extras # DictCursor 사용을 위해 추가
from pgvector.psycopg2 import register_vector # pgvector 등록용
import warnings
import ssl
import urllib.parse
import unicodedata
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

# 2. .env 설정값 로드
MODEL_NAME = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL")
DB_NAME = os.getenv("DB_NAME")

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "dbname": DB_NAME
}

# 설정값 검증
if not all([MODEL_NAME, EMBED_MODEL_NAME, TABLE_NAME, DB_NAME]):
    print("❌ [Error] .env 설정이 누락되었습니다.")
    exit(1)

embedding_lock = threading.Lock()

# 3. 모델 로드
print(f"⏳ [System] 모델 로드 중... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0, timeout=180)
print(f"✅ [System] PostgreSQL pgvector RAG 엔진 가동 준비 완료.")

# --- 유틸리티 함수 ---

def get_internal_context(query: str):
    """PostgreSQL pgvector 전용 <=> 연산자 활용 검색 (형변환 에러 수정)"""
    with embedding_lock:
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = embed_model.encode(instruction + query).tolist()
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn) # 💡 pgvector 타입 등록
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # 💡 [핵심 수정] %s 뒤에 ::vector를 붙여서 명시적으로 형변환을 해줍니다.
        # 이 처리가 없으면 'operator does not exist: vector <=> numeric[]' 에러가 발생합니다.
        sql = f"""
            SELECT original_content, title, source_url, data_source, 
                   (embedding <=> %s::vector) AS dist
            FROM {TABLE_NAME}
            ORDER BY (CASE WHEN data_source = 'feedback' THEN 0 ELSE 1 END) ASC, dist ASC
            LIMIT 5
        """
        cursor.execute(sql, (query_vec,))
        rows = cursor.fetchall()
        
        results = []
        print(f"\n🔍 [Search Debug] 질문: '{query}'")
        for row in rows:
            dist = float(row['dist'])
            print(f"   - [{row['data_source']}] {row['title']} | 거리: {dist:.4f}")
            
            # 임계치 설정 (BGE-Small 기준 0.6 내외 권장)
            if dist < 0.6: 
                source_url = row['source_url'] if row['source_url'] else ""
                normalized = unicodedata.normalize('NFC', source_url)
                encoded = urllib.parse.quote(normalized)
                
                results.append({
                    "content": row['original_content'], 
                    "title": row['title'], 
                    "url": f"{BASE_DOCS_URL}{encoded}" if source_url else "#"
                })
        return results
    except Exception as err:
        print(f"❌ [DB Error] {err}")
        return []
    finally:
        if conn: conn.close()

# --- HTTP 핸들러 ---

class RAGHandler(BaseHTTPRequestHandler):
    def _send_sse(self, data):
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode('utf-8'))
        self.wfile.flush()

    def _send_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == "/feedback":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(content_length))
                query, answer = data.get("query"), data.get("answer")
                if query and answer:
                    self._save_feedback_to_db(query, answer)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "message": "성공적으로 학습되었습니다."}).encode())
            except Exception as e:
                print(f"🚨 피드백 저장 오류: {e}")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/search": return
        
        params = parse_qs(parsed_path.query)
        query = params.get('query', [''])[0]
        if not query: return

        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            self._send_sse({"status": "🔍 지식 데이터 검색 중..."})
            search_results = get_internal_context(query)
            
            if search_results:
                context_combined = "\n".join([f"### [데이터]: {r['content']}" for r in search_results])
                
                prompt = (
                    f"당신은 기술지원 전문가입니다. 반드시 아래 [지식 데이터]만을 근거로 한국어로 답변하세요.\n"
                    f"데이터에 없는 내용은 아는 척하지 마세요.\n\n"
                    f"### [지식 데이터]\n{context_combined}\n\n"
                    f"### [사용자 질문]\n{query}\n\n"
                    f"### [응답 규칙]\n"
                    f"1. [## 📢 조치 안내] -> [### 🛠️ 상세 절차] -> [### 💡 주의 사항] 순서 엄수.\n"
                    f"2. 아주 상세하고 친절하게 답변할 것."
                )
                
                for chunk in llm.stream(prompt):
                    if chunk: self._send_sse({"chunk": chunk})
                
                valid_links = list(set([f"- [{r['title']}]({r['url']})" for r in search_results if r['url'] != "#"]))
                if valid_links:
                    self._send_sse({"chunk": "\n\n---\n### 🔗 관련 문서\n" + "\n".join(valid_links)})
            else:
                self._send_sse({"chunk": "관련 정보를 찾지 못했습니다. 일반 지식으로 답변해 드릴까요?"})

            self._send_done()
        except Exception as e:
            print(f"🚨 서버 오류: {e}")
            self._send_sse({"error": str(e)}); self._send_done()

    def _save_feedback_to_db(self, query, answer):
        combined_text = f"질문: {query}\n전문가 답변: {answer}"
        with embedding_lock:
            # 💡 피드백 저장 시에도 리스트 형태로 전달
            vector = embed_model.encode(combined_text).tolist()
        
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            register_vector(conn)
            cursor = conn.cursor()
            # 💡 여기도 필요시 %s::vector 로 캐스팅할 수 있으나 register_vector가 처리함
            sql = f"""
                INSERT INTO {TABLE_NAME} 
                (content_type, data_source, original_content, embedding, title) 
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, ('text', 'feedback', combined_text, vector, '검증된 답변'))
            conn.commit()
        finally:
            if conn: conn.close()

if __name__ == "__main__":
    # 서버 실행 (localhost:8000)
    server = ThreadingHTTPServer(('0.0.0.0', 8000), RAGHandler)
    print(f"📡 PostgreSQL pgvector RAG Service 가동: http://localhost:8000")
    server.serve_forever()