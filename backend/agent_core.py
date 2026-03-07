import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import psycopg2 
import psycopg2.extras 
from pgvector.psycopg2 import register_vector 
import warnings
import ssl
import urllib.parse
import unicodedata
import threading
import time
import mimetypes 
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote 
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

STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

embedding_lock = threading.Lock()

# 3. 모델 로드
print(f"⏳ [System] 모델 로드 중... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0, timeout=180)
print(f"✅ [System] PostgreSQL 하이브리드 RAG 엔진 가동 준비 완료.")

# --- 유틸리티 함수 ---

def get_internal_context(query: str):
    """
    💡 [최적화] 하이브리드 검색 (벡터 유사도 + 키워드 일치)
    RRF(Reciprocal Rank Fusion) 알고리즘을 사용하여 두 검색 결과의 순위를 통합합니다.
    """
    with embedding_lock:
        # BGE 모델 특성상 지시문 포함
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = embed_model.encode(instruction + query).tolist()
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # 💡 하이브리드 검색 SQL: 벡터 검색과 키워드(to_tsquery) 검색을 결합
        # - 벡터: <=> 연산자 (Cosine Distance)
        # - 키워드: ts_rank (키워드 빈도 및 근접도 점수)
        sql = f"""
            WITH vector_matches AS (
                SELECT id, (embedding <=> %s::vector) AS dist
                FROM {TABLE_NAME}
                ORDER BY dist ASC
                LIMIT 20
            ),
            keyword_matches AS (
                SELECT id, ts_rank(content_search_vector, plainto_tsquery('simple', %s)) AS rank
                FROM {TABLE_NAME}
                WHERE content_search_vector @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT 20
            )
            SELECT 
                t.original_content, t.title, t.source_url, t.data_source,
                COALESCE(1.0 / (60 + v.dist * 100), 0) + COALESCE(k.rank, 0) AS combined_score
            FROM {TABLE_NAME} t
            LEFT JOIN vector_matches v ON t.id = v.id
            LEFT JOIN keyword_matches k ON t.id = k.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL
            ORDER BY (CASE WHEN t.data_source = 'feedback' THEN 0 ELSE 1 END) ASC, combined_score DESC
            LIMIT 7;
        """
        
        # plainto_tsquery를 사용해 일반 문장을 검색어 쿼리로 변환합니다.
        cursor.execute(sql, (query_vec, query, query))
        rows = cursor.fetchall()
        
        results = []
        print(f"\n🔍 [Hybrid Search Debug] 질문: '{query}'")
        for row in rows:
            print(f"   - [{row['data_source']}] {row['title']} | 점수: {row['combined_score']:.4f}")
            
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

    def serve_static_file(self, file_path):
        try:
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
        
        if parsed_path.path.startswith("/files/"):
            file_path = parsed_path.path[7:]
            self.serve_static_file(file_path)
            return

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
        """
        💡 [피드백 저장 시 하이브리드 검색 컬럼 대응]
        """
        combined_text = f"질문: {query}\n전문가 답변: {answer}"
        with embedding_lock:
            vector = embed_model.encode(combined_text).tolist()
        
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            register_vector(conn)
            cursor = conn.cursor()
            # 💡 content_search_vector(tsvector)를 함께 저장
            sql = f"""
                INSERT INTO {TABLE_NAME} 
                (content_type, data_source, original_content, embedding, title, content_search_vector) 
                VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """
            cursor.execute(sql, ('text', 'feedback', combined_text, vector, '검증된 답변', combined_text))
            conn.commit()
        finally:
            if conn: conn.close()

if __name__ == "__main__":
    server = ThreadingHTTPServer(('0.0.0.0', 8000), RAGHandler)
    print(f"📡 하이브리드 RAG Service 가동: http://localhost:8000")
    server.serve_forever()