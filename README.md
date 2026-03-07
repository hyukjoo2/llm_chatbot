# 🚀 AI Agent

## 🛠️ 설치 및 실행 방법

### 설정 및 배포 (Python + Next.js)
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치 (중요!)
pip install -r requirements.txt

# 서버 실행
python agent_core.py
```ㅁ

### 2. 프론트엔드 설정 (Next.js)

``` bash
# 라이브러리 설치
npm install

# 개발 서버 실행
npm run dev

# 배포
npm run build

npm run start

# change port
npm run start -- -p 80

# [실전 팁] 서버가 꺼지지 않게 하려면? (PM2 사용)
## PM2 설치:
npm install -g pm2

## Next.js 프로젝트 PM2로 실행:
# 프로젝트 루트 폴더에서 실행
pm2 start npm --name "my-rag-chatbot" -- start

## 상태 확인 및 관리:
pm2 status   # 현재 실행 중인 리스트 확인
pm2 logs     # 실시간 로그 확인
pm2 restart my-rag-chatbot  # 재시작

## 3. [주의 사항] 환경 변수(.env) 관리
배포 시 가장 많이 실수하는 부분입니다.
API 주소: 개발 때는 localhost:8000이었지만, 실제 배포 시 파이썬 백엔드 주소가 바뀐다면 .env를 해당 도메인이나 고정 IP로 수정하고 **다시 빌드(npm run build)**해야 합니다.

## 4. [배포 방식] 어디에 배포하시나요?
현재 어떤 환경에 배포하실 계획인가요? 환경에 따라 방법이 살짝 다릅니다.
Vercel (가장 추천): Next.js 만든 회사에서 운영하며, 깃허브 연결만 하면 자동으로 빌드부터 배포까지 끝내줍니다. (무료 플랜 존재)
개인 서버 (Ubuntu/AWS 등): 위에서 설명한 npm run build + PM2 조합으로 직접 운영합니다.
Docker: Dockerfile을 만들어 컨테이너로 배포합니다.
```

### 3. DB 접속
```
docker exec -it pg-vector psql -U myuser -d llm_chatbot

TRUNCATE TABLE rag_vectors;
```
