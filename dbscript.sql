-- 1. 데이터베이스 생성 (이미 있다면 생략 가능)
-- 주의: psql 접속 상태에서 실행하거나, 생성 후 \c llm_chatbot으로 이동해야 합니다.
CREATE DATABASE llm_chatbot;

-- 2. 해당 데이터베이스로 접속
\c llm_chatbot

-- 3. pgvector 확장 모듈 활성화 (PostgreSQL에서 벡터 기능을 쓰기 위해 필수)
CREATE EXTENSION IF NOT EXISTS vector;

-- 4. 기존 테이블 삭제 (깔끔하게 새로 시작하고 싶을 때 실행)
DROP TABLE IF EXISTS rag_vectors;

-- 5. 하이브리드 검색 전용 테이블 생성
CREATE TABLE rag_vectors (
    id SERIAL PRIMARY KEY,
    
    -- 데이터 성격 및 출처 구분
    content_type VARCHAR(10) NOT NULL CHECK (content_type IN ('text', 'image')),
    data_source VARCHAR(10) DEFAULT 'manual' CHECK (data_source IN ('manual', 'feedback')),
    
    -- [핵심] 원문 데이터
    original_content TEXT NOT NULL, 
    
    -- [벡터] BGE-Small (384차원)
    embedding vector(384) NOT NULL, 
    
    -- [키워드 검색용] 원문을 검색 엔진용 포맷으로 저장하는 컬럼
    content_search_vector tsvector,

    -- [출처 및 메타정보]
    title VARCHAR(255),
    source_url TEXT,
    feedback_score INT DEFAULT 0,
    metadata JSONB, 
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- --- 인덱스 설정 (검색 속도 최적화) ---

-- 1. 벡터 검색용 인덱스 (HNSW 방식: 속도가 매우 빠름)
CREATE INDEX idx_rag_vectors_embedding ON rag_vectors USING hnsw (embedding vector_cosine_ops);

-- 2. 키워드 전문 검색용 인덱스 (GIN 인덱스)
CREATE INDEX idx_rag_vectors_search ON rag_vectors USING gin(content_search_vector);

-- 3. 필터링용 일반 인덱스
CREATE INDEX idx_content_type ON rag_vectors (content_type);
CREATE INDEX idx_data_source ON rag_vectors (data_source);