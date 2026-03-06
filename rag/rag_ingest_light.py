import os
import json
import argparse
import psycopg2  # MySQL 대신 PostgreSQL용 라이브러리
from pgvector.psycopg2 import register_vector  # 벡터 타입 등록용
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 환경 변수 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "..", "backend", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"📡 설정 로드 완료: {dotenv_path}")
else:
    load_dotenv()
    print("⚠️ .env 파일을 찾을 수 없어 기본 환경변수를 사용합니다.")

class RAGIngestionLight:
    def __init__(self):
        print("🤖 라이트 에이전트 모델 로딩 중 (BGE-Small)...")
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       
            chunk_overlap=100,     
            length_function=len,
            separators=["\n\n", "\n", ". ", "!", "?", " ", ""]
        )
        print("✅ 가벼운 임베딩 모델 및 청킹 설정 완료")

    def _get_db_conn(self):
        # PostgreSQL 접속 설정
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        # 💡 pgvector 타입을 psycopg2에서 인식하도록 등록 (매우 중요)
        register_vector(conn)
        return conn

    def _save_to_db(self, cursor, c_type, content, f_name, page_info):
        if not content or len(content) < 15: return
        
        # 1. BGE-Small (384차원) 임베딩 생성 (리스트 형태 유지)
        embedding = self.embed_model.encode(content).tolist()
        
        # 2. 파일명 처리
        if f_name.lower().endswith(".txt"):
            display_name = f_name[:-4]
        else:
            display_name = f_name
            
        metadata = json.dumps({
            "source_type": c_type,
            "page_info": page_info, 
            "ingested_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }, ensure_ascii=False)
        
        table_name = os.getenv('DB_TABLE_NAME', 'rag_vectors')
        
        # 3. PostgreSQL INSERT (pgvector 방식)
        # %s 자리에 임베딩 리스트를 그대로 넣으면 됩니다.
        sql = f"""
            INSERT INTO {table_name} 
            (content_type, original_content, embedding, title, source_url, metadata) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(sql, (
            c_type, 
            content, 
            embedding, 
            display_name,
            display_name,
            metadata
        ))

    def ingest_txt_file(self, file_path):
        if not file_path.lower().endswith('.txt'): return
        print(f"🚀 벡터화 및 DB 저장 시작: {file_path}")
        
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_content = f.read()

            conn = self._get_db_conn()
            cursor = conn.cursor()

            if len(full_content) > 20:
                chunks = self.text_splitter.split_text(full_content)
                for i, chunk in enumerate(chunks):
                    c_type = 'image' if '[이미지 요약]' in chunk or '[OCR 추출]' in chunk else 'text'
                    self._save_to_db(cursor, c_type, chunk, file_name, f"Chunk-{i+1}")
            
            conn.commit()
            saved_name = file_name[:-4] if file_name.lower().endswith(".txt") else file_name
            print(f"✅ DB 저장 성공: {saved_name}")
            cursor.close(); conn.close()
        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")

def main():
    agent = RAGIngestionLight()
    # 텍스트 파일 경로 설정
    text_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converted_texts")
    
    if not os.path.exists(text_dir):
        print(f"⚠️ '{text_dir}' 폴더가 없습니다.")
        return

    files = [f for f in os.listdir(text_dir) if f.lower().endswith(".txt")]
    if not files:
        print("📁 변환된 텍스트 파일이 없습니다.")
        return

    for f in files:
        agent.ingest_txt_file(os.path.join(text_dir, f))
    
    print("\n✨ 모든 작업이 완료되었습니다. 이제 검색 쿼리를 날려보세요!")

if __name__ == "__main__":
    main()