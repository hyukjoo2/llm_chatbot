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
import mimetypes # 💡 파일 타입 인식을 위해 추가
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote # 💡 unquote 추가
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
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL") # 💡 .env에서 http://localhost:8000/files/ 로 수정 필요
DB_NAME = os.getenv("DB_NAME")

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "dbname": DB_NAME
}

# 💡 실시간 파일 저장용 폴더 생성
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

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
    """PostgreSQL pgvector 전용 <=> 연산자 활용 검색"""
    with embedding_lock:
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = embed_model.encode(instruction + query).tolist()
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # 💡 LIMIT를 10으로 늘려 추론 재료를 더 많이 확보합니다.
        sql = f"""
            SELECT original_content, title, source_url, data_source, 
                   (embedding <=> %s::vector) AS dist
            FROM {TABLE_NAME}
            ORDER BY (CASE WHEN data_source = 'feedback' THEN 0 ELSE 1 END) ASC, dist ASC
            LIMIT 10
        """
        cursor.execute(sql, (query_vec,))
        rows = cursor.fetchall()
        
        results = []
        print(f"\n🔍 [Search Debug] 질문: '{query}'")
        for row in rows:
            dist = float(row['dist'])
            print(f"   - [{row['data_source']}] {row['title']} | 거리: {dist:.4f}")
            
            if dist < 0.6: 
                source_url = row['source_url'] if row['source_url'] else ""
                # 💡 한글 자소 분리 방지 및 인코딩
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

    # 💡 정적 파일 서빙 로직 (운영 중 실시간 파일 조회용)
    def serve_static_file(self, file_path):
        try:
            # URL 인코딩된 경로 해제 (한글 포함)
            decoded_filename = unquote(file_path)
            full_path = os.path.join(STORAGE_DIR, decoded_filename)

            if os.path.exists(full_path) and os.path.isfile(full_path):
                self.send_response(200)
                mime_type, _ = mimetypes.guess_type(full_path)
                self.send_header('Content-type', mime_type or 'application/octet-stream')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(full_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File Not Found")
        except Exception as e:
            self.send_error(500, str(e))

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
        
        # 💡 1. 파일 요청 처리 (/files/...)
        if parsed_path.path.startswith("/files/"):
            file_path = parsed_path.path[7:] # '/files/' 이후 부분
            self.serve_static_file(file_path)
            return

        # 2. 검색 요청 처리 (/search)
        if parsed_path.path == "/search":
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
                    
                    # 💡 [논리적 추론 허용 프롬프트]로 수정
                    prompt = (
                        f"당신은 기술지원 전문가입니다. 반드시 제공된 [지식 데이터]를 근거로 답변하되, "
                        f"직접적인 절차가 없더라도 데이터 내의 메뉴명, 버튼 이름 등을 활용해 논리적으로 추론하여 안내하세요.\n"
                        f"단, 데이터에 전혀 근거가 없는 내용은 지어내지 마세요.\n\n"
                        f"### [지식 데이터]\n{context_combined}\n\n"
                        f"### [사용자 질문]\n{query}\n\n"
                        f"### [응답 규칙]\n"
                        f"1. [## 📢 조치 안내] -> [### 🛠️ 상세 절차] -> [### 💡 주의 사항] 순서 엄수.\n"
                        f"2. 추론한 내용일 경우 '매뉴얼 기반 추론 절차입니다'라고 명시하세요.\n"
                        f"3. 아주 상세하고 친절하게 답변할 것."
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
            vector = embed_model.encode(combined_text).tolist()
        
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            register_vector(conn)
            cursor = conn.cursor()
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
    print(f"📂 정적 파일 저장소: {STORAGE_DIR}")
    server.serve_forever()