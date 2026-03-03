CREATE DATABASE IF NOT EXISTS llm_chatbot;
USE llm_chatbot;

CREATE TABLE rag_vectors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- [데이터 성격 구분] 텍스트인지 이미지(OCR)인지 구분
    content_type ENUM('text', 'image') NOT NULL,

    -- 🚨 [추가] 데이터의 출처 구분: 'manual'(기본 지식), 'feedback'(사용자가 추천한 답변)
    data_source ENUM('manual', 'feedback') DEFAULT 'manual',
    
    -- [핵심 데이터] 원문, 이미지 요약 또는 검증된 Q&A 답변
    original_content TEXT NOT NULL, 
    
    -- [벡터 데이터] BGE-M3 (1024차원) 임베딩 저장
    embedding VECTOR(1024) NOT NULL, 
    
    -- [출처 정보] 
    -- 문서 제목 (feedback일 경우 "사용자 검증 답변" 등으로 저장)
    title VARCHAR(255),
    -- 실제 참조 가능한 URL 또는 서버 내 파일 경로
    source_url TEXT,
    
    -- 🚨 [추가] 피드백 점수 저장 (기본 0, 따봉 시 +1 등 활용 가능)
    feedback_score INT DEFAULT 0,
    
    -- [확장용] 페이지 번호, 작성자, 태그 등 기타 정보
    metadata JSON, 
    
    -- [관리용]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 검색 최적화를 위한 인덱스
    INDEX idx_content_type (content_type),
    INDEX idx_data_source (data_source) -- 출처별 필터링을 위한 인덱스 추가
);