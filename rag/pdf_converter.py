import os
import io
import fitz  # PyMuPDF
import re
import easyocr
import unicodedata
from PIL import Image

class PDFToTextConverter:
    def __init__(self):
        print("🚀 [Step 1] 분석 모델 로딩 중 (EasyOCR)...")
        # 💡 BLIP(이미지 요약) 모델을 제거하여 '카펫' 오진을 원천 차단합니다.
        # 한국어와 영어를 동시에 인식하도록 설정합니다.
        self.reader = easyocr.Reader(['ko', 'en'])
        print("✅ 분석 모델 로드 완료")

    def _clean_text(self, text):
        """텍스트 정제: 불필요한 공백 제거 및 유니코드 정규화"""
        if not text: return ""
        # 맥북 한글 자소 분리 방지를 위한 NFC 정규화
        text = unicodedata.normalize('NFC', text)
        # 연속된 공백 및 줄바꿈 정리
        text = re.sub(r'\s+', ' ', text)
        # 너무 짧은 텍스트는 유의미하지 않으므로 필터링 (최소 5자)
        return text.strip() if len(text.strip()) >= 5 else ""

    def _analyze_image_with_ocr(self, image_bytes):
        """이미지 내의 글자만 OCR로 추출 (이미지 요약 기능 제외)"""
        try:
            # 💡 BLIP 대신 OCR만 사용하여 이미지 속 메뉴명, 버튼명 등을 정확히 읽어냅니다.
            ocr_results = self.reader.readtext(image_bytes, detail=0)
            ocr_text = " ".join(ocr_results)
            # OCR 추출 결과가 있을 때만 반환
            return f"\n[이미지 내 텍스트]: {ocr_text}\n" if ocr_text.strip() else ""
        except Exception as e:
            print(f"⚠️ OCR 오류 발생: {e}")
            return ""

    def convert(self, file_path):
        doc = fitz.open(file_path)
        file_name = os.path.basename(file_path)
        output_data = []

        print(f"📖 {file_name} 변환 시작 (총 {len(doc)} 페이지)...")
        for page_num, page in enumerate(doc):
            # 1. 텍스트 레이어 추출 (가장 정확한 본문 추출 방식)
            content = page.get_text("text")
            cleaned = self._clean_text(content)
            
            page_entry = f"--- Page {page_num + 1} ---\n{cleaned}\n"
            
            # 2. 페이지 내 포함된 이미지들에서 OCR 추출
            # 💡 매뉴얼 스크린샷 내의 텍스트를 잡기 위함
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    ocr_content = self._analyze_image_with_ocr(image_bytes)
                    if ocr_content:
                        page_entry += ocr_content
                except Exception as e:
                    continue
            
            output_data.append(page_entry)

        # 3. 결과 저장
        output_dir = "converted_texts"
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_data))
        
        doc.close()
        print(f"✅ 변환 완료: {output_path}")

if __name__ == "__main__":
    converter = PDFToTextConverter()
    
    # 💡 PDF 파일이 있는 소스 폴더 경로 (상황에 맞게 수정)
    source_dir = "./sources" 
    
    if not os.path.exists(source_dir):
        print(f"⚠️ '{source_dir}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 PDF를 넣어주세요.")
    else:
        pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"📁 '{source_dir}' 폴더에 PDF 파일이 없습니다.")
        else:
            for f in pdf_files:
                converter.convert(os.path.join(source_dir, f))
    
    print("\n✨ 모든 PDF 변환 작업이 끝났습니다. 'converted_texts' 폴더를 확인하세요!")