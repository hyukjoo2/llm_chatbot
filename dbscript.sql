CREATE DATABASE IF NOT EXISTS llm_chatbot;
\c llm_chatbot

-- 기존 테이블이 있다면 삭제 (데이터가 없다면 실행)
DROP TABLE IF EXISTS rag_vectors;

CREATE TABLE rag_vectors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- [데이터 성격 구분] 텍스트인지 이미지(OCR)인지 구분
    content_type ENUM('text', 'image') NOT NULL,

    -- 데이터의 출처 구분: 'manual'(기본 지식), 'feedback'(사용자가 추천한 답변)
    data_source ENUM('manual', 'feedback') DEFAULT 'manual',
    
    -- [핵심 데이터] 원문, 이미지 요약 또는 검증된 Q&A 답변
    original_content TEXT NOT NULL, 
    
    -- 🚨 MariaDB 11.4 내장 벡터 (384차원)
    embedding VECTOR(384) NOT NULL, 
    
    -- [출처 정보] 원본 .pdf 파일명이 저장될 곳 (사용자님이 말씀하신 '파일 이름')
    title VARCHAR(255),
    source_url TEXT,
    
    -- 피드백 점수 (추후 검색 가중치 활용 가능)
    feedback_score INT DEFAULT 0,
    
    -- [확장용] 페이지 번호, 인제스트 날짜 등
    metadata JSON, 
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 검색 속도 최적화를 위한 인덱스
    INDEX idx_content_type (content_type),
    INDEX idx_data_source (data_source)
);