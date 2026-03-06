import os
import io
import json
import fitz
import re
from PIL import Image
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration

class PDFToTextConverter:
    def __init__(self):
        print("🚀 [Step 1] BGE-M3 및 분석 모델 로딩 중...")
        # 임베딩은 하지 않지만, 나중에 M3의 시각적 분석 능력을 확장할 수 있으므로 
        # 추출용도로만 모델 세팅 (현재는 OCR/BLIP 중심)
        model_id = "Salesforce/blip-image-captioning-base"
        self.caption_processor = BlipProcessor.from_pretrained(model_id)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(model_id)
        self.reader = easyocr.Reader(['ko', 'en'])
        print("✅ 분석 모델 로드 완료")

    def _clean_text(self, text):
        if not text: return ""
        text = "".join(ch for ch in text if ch.isprintable())
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!\'\"()\-\[\]]', '', text)
        return text.strip() if len(text.strip()) >= 10 else ""

    def _analyze_image(self, image_bytes):
        try:
            raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.caption_processor(raw_image, return_tensors="pt")
            out = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            ocr_text = " ".join(self.reader.readtext(image_bytes, detail=0))
            return f"\n[이미지 요약]: {caption}\n[OCR 추출]: {ocr_text}\n"
        except: return ""

    def convert(self, file_path):
        doc = fitz.open(file_path)
        file_name = os.path.basename(file_path)
        output_data = []

        print(f"📖 {file_name} 변환 시작...")
        for page_num, page in enumerate(doc):
            page_text = page.get_text("blocks")
            content = " ".join([b[4] for b in page_text if b[4]])
            cleaned = self._clean_text(content)
            
            # 페이지별 텍스트 추가
            page_entry = f"--- Page {page_num + 1} ---\n{cleaned}\n"
            
            # 이미지 분석 추가
            for img_info in page.get_images(full=True):
                img_data = doc.extract_image(img_info[0])["image"]
                page_entry += self._analyze_image(img_data)
            
            output_data.append(page_entry)

        # 결과 저장
        output_dir = "converted_texts"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_data))
        
        doc.close()
        print(f"✅ 변환 완료: {output_path}")

if __name__ == "__main__":
    converter = PDFToTextConverter()
    # sources 폴더 내 pdf 처리
    source_dir = "./sources"
    for f in os.listdir(source_dir):
        if f.lower().endswith(".pdf"):
            converter.convert(os.path.join(source_dir, f))