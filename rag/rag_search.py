import os
import json
import numpy as np
import mysql.connector
import warnings
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. SSL 관련 경고 무시 및 환경 변수 로드
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
load_dotenv()

class RAGSearcher:
    def __init__(self):
        # 💡 [수정] .env 설정에 맞춰 BGE-Small 로드
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        print(f"🔍 검색 엔진 로딩 중 ({model_name})...")
        self.embed_model = SentenceTransformer(model_name)
        print("✅ 검색 준비 완료!")

    def _get_db_conn(self):
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "3306"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            auth_plugin='mysql_native_password'
        )

    def search(self, user_query, top_k=3):
        # 1. 질문을 384차원 벡터로 변환
        query_vec = self.embed_model.encode(user_query).tolist()
        query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"
        
        conn = self._get_db_conn()
        cursor = conn.cursor(dictionary=True)

        try:
            # 💡 [핵심 수정] 파이썬에서 계산하지 않고 MySQL 엔진에서 직접 벡터 연산 수행
            # VECTOR_DISTANCE(L2_SQUARED) 또는 코사인 유사도 연산 사용
            table_name = os.getenv('DB_TABLE_NAME')
            
            # MySQL 9.x 이상에서 지원하는 벡터 거리 연산 쿼리
            # 거리가 짧을수록(유사도가 높을수록) 상단에 노출
            sql = f"""
                SELECT 
                    id, content_type, original_content, title, source_url, metadata,
                    VECTOR_DISTANCE(embedding, STRING_TO_VECTOR(%s)) as distance
                FROM {table_name}
                ORDER BY distance ASC
                LIMIT %s
            """
            
            cursor.execute(sql, (query_vec_str, top_k))
            results = cursor.fetchall()
            return results

        except Exception as e:
            print(f"❌ 검색 도중 오류 발생: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Vector Search Tool")
    parser.add_argument("--query", type=str, required=True, help="검색할 질문")
    parser.add_argument("--top", type=int, default=3, help="출력할 결과 개수")
    args = parser.parse_args()

    searcher = RAGSearcher()
    results = searcher.search(args.query, top_k=args.top)

    print(f"\n#️⃣ '{args.query}' 검색 결과:\n" + "="*50)
    
    if not results:
        print("검색 결과가 없습니다. DB 인제스트 여부를 확인하세요.")
    else:
        for i, res in enumerate(results, 1):
            print(f"[{i}] 거리(유사도 점수): {res['distance']:.4f}")
            print(f"📍 출처: {res.get('title', 'N/A')} (URL: {res.get('source_url', 'N/A')})")
            print(f"📝 내용 요약: {res['original_content'][:200]}...") # 너무 길면 요약 출력
            print("-" * 50)