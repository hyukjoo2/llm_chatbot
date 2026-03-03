import os

# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import mysql.connector
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

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
embedding_lock = threading.Lock()

# 2. 설정값
MODEL_NAME = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD", ""), 
    "database": os.getenv("DB_NAME")
}
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL", "http://localhost:3000/sources/")

chat_histories = {}

# 3. 모델 로드
print(f"⏳ [System] 모델 로드 중... ({MODEL_NAME})")
embed_model = SentenceTransformer('BAAI/bge-m3')
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0, timeout=180)
print(f"✅ [System] 로드 완료.")

def get_internal_context(query: str):
    """DB 지식 검색 (피드백 데이터 우선순위 적용)"""
    with embedding_lock:
        query_vec = embed_model.encode(query)
    
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        # MySQL 9 VECTOR_TO_STRING 활용
        sql = f"""
            SELECT original_content, title, source_url, data_source, VECTOR_TO_STRING(embedding) as vector_str 
            FROM {TABLE_NAME}
            ORDER BY data_source DESC
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            if not row['vector_str']: continue
            db_vec = np.array(json.loads(row['vector_str']))
            distance = np.linalg.norm(query_vec - db_vec)
            
            # 유사도 임계치 2.6
            if distance < 2.6:
                source_url = row['source_url'] if row['source_url'] else ""
                normalized = unicodedata.normalize('NFC', source_url)
                encoded = urllib.parse.quote(normalized)
                
                results.append({
                    "distance": distance, 
                    "content": row['original_content'], 
                    "title": row['title'], 
                    "data_source": row['data_source'],
                    "url": f"{BASE_DOCS_URL}{encoded}" if source_url else "#"
                })
        
        # 피드백 데이터를 최우선(0), 매뉴얼을 다음(1)으로 정렬 후 거리순 정렬
        results.sort(key=lambda x: (0 if x['data_source'] == 'feedback' else 1, x['distance']))
        return results[:3] 
    except Exception as e:
        print(f"❌ DB 검색 오류: {e}")
        return []
    finally:
        if conn and conn.is_connected(): conn.close()

# --- HTTP 핸들러 ---

class RAGHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_OPTIONS(self):
        """브라우저 CORS 대응 (Preflight)"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """피드백 저장 엔드포인트"""
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/feedback":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                
                query = data.get("query")
                answer = data.get("answer")
                
                if query and answer:
                    self._save_feedback_to_db(query, answer)
                    self._send_json_response({"status": "success", "message": "학습이 완료되었습니다."})
                else:
                    self._send_json_response({"status": "error", "message": "데이터가 부족합니다."}, status=400)
            except Exception as e:
                print(f"🚨 [POST Error] {e}")
                self._send_json_response({"status": "error", "message": str(e)}, status=500)

    def do_GET(self):
        """질문 답변 생성 (SSE 스트리밍)"""
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/search": return
        params = parse_qs(parsed_path.query)
        query = params.get('query', [''])[0]
        session_id = params.get('sessionId', ['default_session'])[0]
        if not query: return

        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            if session_id not in chat_histories: chat_histories[session_id] = []
            
            self._send_sse({"status": "🧠 지식 데이터 분석 중..."})
            search_results = get_internal_context(query)
            
            full_reply = ""
            if search_results:
                context_combined = "\n".join([
                    f"[{'기존 매뉴얼' if r['data_source']=='manual' else '검증된 답변'} {i+1}]:\n{r['content']}\n" 
                    for i, r in enumerate(search_results)
                ])
                
                # 수정된 프롬프트 예시
                prompt = (
                    f"당신은 기술 지원 시니어 엔지니어입니다. 제공된 [지식 데이터]를 분석하여 답변하세요.\n\n"
                    f"### [지식 데이터]\n{context_combined}\n\n"
                    f"### [사용자 질문]\n{query}\n\n"
                    f"### [응답 규칙]\n"
                    f"1. **중요**: 질문이 [지식 데이터]의 내용과 관련이 없다면, 절대 지식 데이터를 언급하지 말고 일반적인 답변만 하세요.\n"  # 🚨 이 줄을 추가하세요!
                    f"2. 직접 서술: 문서 참조 번호는 생략하고 내용을 상세히 풀어서 설명하세요.\n"
                    f"3. 구조: [## 📢 조치 안내] -> [### 🛠️ 상세 절차] -> [### 💡 주의 사항] 순서로 작성하세요."
                )
                
                for chunk in llm.stream(prompt):
                    if chunk: 
                        self._send_sse({"chunk": chunk}); full_reply += chunk
                
                valid_links = [f"- [{r['title']}]({r['url']})" for r in search_results if r['data_source'] == 'manual' and r['url'] != "#"]
                if valid_links:
                    source_section = "\n\n---\n### 🔗 참고 문서\n" + "\n".join(list(set(valid_links)))
                    self._send_sse({"chunk": source_section}); full_reply += source_section
            else:
                for chunk in llm.stream(f"친절한 기술 상담원으로서 답변하세요: {query}"):
                    if chunk: self._send_sse({"chunk": chunk}); full_reply += chunk

            chat_histories[session_id].append({"role": "user", "content": query})
            chat_histories[session_id].append({"role": "assistant", "content": full_reply})
            if len(chat_histories[session_id]) > 4: chat_histories[session_id].pop(0)

            self._send_done()

        except Exception as e:
            print(f"🚨 [GET Error] {e}")
            self._send_sse({"error": str(e)}); self._send_done()

    def _save_feedback_to_db(self, query, answer):
        """MySQL 9 전용 피드백 저장 로직 (STRING_TO_VECTOR 및 직접 캐스팅 대응)"""
        combined_text = f"질문: {query}\n전문가 답변: {answer}"
        
        with embedding_lock:
            vector_np = embed_model.encode(combined_text)
            vector_list = vector_np.tolist()
        
        # MySQL 9 벡터 규격 문자열 포맷팅
        vector_str = "[" + ",".join(map(str, vector_list)) + "]"
        
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            # 1. STRING_TO_VECTOR 함수 사용 시도 (MySQL 9 정석)
            try:
                sql = f"""
                    INSERT INTO {TABLE_NAME} 
                    (content_type, data_source, original_content, embedding, title, feedback_score)
                    VALUES (%s, %s, %s, STRING_TO_VECTOR(%s), %s, %s)
                """
                params = ('text', 'feedback', combined_text, vector_str, '현장 검증 답변', 1)
                cursor.execute(sql, params)
            except mysql.connector.Error as err:
                if err.errno == 1305: # 함수가 없는 경우 직접 캐스팅 시도
                    sql = f"""
                        INSERT INTO {TABLE_NAME} 
                        (content_type, data_source, original_content, embedding, title, feedback_score)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, params)
                else: raise err

            conn.commit()
            print(f"✅ [Learning] 현장 지식 저장 성공: {query[:15]}...")
            
        except mysql.connector.Error as err:
            print(f"❌ MySQL 저장 실패: {err}")
            raise err
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def _send_sse(self, data):
        try:
            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode('utf-8'))
            self.wfile.flush()
        except: pass

    def _send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_done(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except: pass

if __name__ == "__main__":
    server = ThreadingHTTPServer(('localhost', 8000), RAGHandler)
    print(f"📡 Intelligent Learning Agent 가동: http://localhost:8000")
    server.serve_forever()