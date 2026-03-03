import os

# 🚨 [중요] Mac 환경에서 Segmentation Fault 방지를 위해 최상단에 배치
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import mysql.connector
import warnings
import ssl
import urllib.parse
import unicodedata
import threading  # 🚨 동기화 Lock을 위해 추가
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 1. 환경 변수 로드
load_dotenv()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

# 경고 무시
warnings.filterwarnings("ignore")

# 🚨 [해결책] 임베딩 모델 충돌 방지용 Lock 선언
# 여러 스레드가 동시에 embed_model을 호출하지 못하도록 순서를 정해줍니다.
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
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
print(f"✅ [System] 로드 완료.")

# --- 핵심 로직 ---

def analyze_request(query: str, history_text: str = ""):
    """의도 분류와 질문 분해"""
    prompt = (
        f"이전 대화 맥락:\n{history_text}\n\n"
        f"사용자 질문: '{query}'\n\n"
        "당신은 분석 에이전트입니다. 반드시 JSON 형식으로만 답변하세요. 예: {\"intent\": \"INFO\", \"keywords\": [\"키워드1\"]}"
    )
    try:
        res = llm.invoke(prompt).strip()
        clean_res = res.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        return data
    except:
        return {"intent": "INFO", "keywords": [query]}

def get_internal_context(query: str):
    """DB 지식 검색 로직 (Thread-Safe + Lock 적용)"""
    
    # 🚨 [핵심 수정] 임베딩 생성 시점에만 Lock을 걸어 세그먼테이션 폴트 방지
    with embedding_lock:
        query_vec = embed_model.encode(query)
    
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        sql = f"SELECT original_content, title, source_url, VECTOR_TO_STRING(embedding) as vector_str FROM {TABLE_NAME}"
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            if not row['vector_str']: continue
            db_vec = np.array(json.loads(row['vector_str']))
            distance = np.linalg.norm(query_vec - db_vec)
            
            if distance < 2.8:
                file_name = row['source_url']
                normalized_file_name = unicodedata.normalize('NFC', file_name)
                encoded_file_name = urllib.parse.quote(normalized_file_name)
                full_url = f"{BASE_DOCS_URL}{encoded_file_name}" if file_name else ""
                
                results.append({
                    "distance": distance, "content": row['original_content'], "title": row['title'], "url": full_url
                })
        results.sort(key=lambda x: x['distance'])
        return results[:3] 
    except Exception as e:
        print(f"⚠️ DB 검색 에러: {e}")
        return []
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def parallel_search_and_merge(keywords):
    """안정성이 강화된 병렬 검색"""
    if not keywords: return []

    search_tasks = {
        f"task_{i}": RunnableLambda(lambda x, k=kw: get_internal_context(k))
        for i, kw in enumerate(keywords)
    }

    parallel_chain = RunnableParallel(**search_tasks)
    
    try:
        # Lock이 보호해주므로 max_concurrency는 유지해도 안전합니다.
        raw_results = parallel_chain.invoke({}, config={"max_concurrency": 3})
    except Exception as e:
        print(f"⚠️ 병렬 실행 충돌 발생, 순차 처리 전환")
        raw_results = {f"task_{i}": get_internal_context(k) for i, k in enumerate(keywords)}

    all_res = []
    seen_content = set()
    for task_id in raw_results:
        for item in raw_results[task_id]:
            if item['content'] not in seen_content:
                all_res.append(item)
                seen_content.add(item['content'])
    return all_res

# --- HTTP 핸들러 ---

class RAGHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/search": return
        
        params = parse_qs(parsed_path.query)
        query = params.get('query', [''])[0]
        session_id = params.get('sessionId', ['default_session'])[0]
        
        if not query: return

        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            if session_id not in chat_histories: chat_histories[session_id] = []
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in chat_histories[session_id][-4:]])

            self._send_sse({"status": "🧠 분석 중..."})
            analysis = analyze_request(query, history_text)
            intent = analysis.get("intent", "INFO").upper()
            keywords = analysis.get("keywords", [query])
            
            full_assistant_reply = ""

            if intent == "CHAT":
                self._send_sse({"status": "💬 답변 생성 중..."})
                for chunk in llm.stream(f"기술 지원 엔지니어로서 답변하세요: {query}"):
                    if chunk: 
                        self._send_sse({"chunk": chunk}); full_assistant_reply += chunk
            else:
                self._send_sse({"status": f"🔍 지식 검색 중 ({len(keywords)}개)..."})
                search_results = parallel_search_and_merge(keywords)
                
                if search_results:
                    self._send_sse({"status": "📋 가이드 작성 중..."})
                    context_text = "\n".join([f"- {r['content']}" for r in search_results])
                    source_links = []
                    seen_urls = set()
                    for r in search_results:
                        if r['url'] and r['url'] not in seen_urls:
                            source_links.append(f"- [{r['title']}]({r['url']})")
                            seen_urls.add(r['url'])
                    
                    prompt = f"### [지식 데이터]\n{context_text}\n\n### [질문]\n{query}"
                    for chunk in llm.stream(prompt):
                        if chunk: 
                            self._send_sse({"chunk": chunk}); full_assistant_reply += chunk
                    
                    if source_links:
                        source_section = "\n\n---\n### 🔗 참고 문서\n" + "\n".join(source_links)
                        self._send_sse({"chunk": source_section}); full_assistant_reply += source_section
                else:
                    for chunk in llm.stream(query):
                        if chunk: 
                            self._send_sse({"chunk": chunk}); full_assistant_reply += chunk

            chat_histories[session_id].append({"role": "user", "content": query})
            chat_histories[session_id].append({"role": "assistant", "content": full_assistant_reply})
            self._send_done()

        except Exception as e:
            self._send_sse({"error": str(e)})
            try: self._send_done()
            except: pass

    def _send_sse(self, data):
        payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(payload.encode('utf-8'))
        self.wfile.flush()

    def _send_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), RAGHandler)
    print(f"📡 Mac Optimized Agent 가동: http://localhost:8000")
    server.serve_forever()