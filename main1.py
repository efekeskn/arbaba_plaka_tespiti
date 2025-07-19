import cv2
import numpy as np
import easyocr
import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlateCandidate:
    """Plaka adayı bilgilerini tutan veri sınıfı"""
    region: np.ndarray
    coordinates: Tuple[int, int, int, int]
    confidence: float
    area: int
    aspect_ratio: float
    method: str

@dataclass
class PlateResult:
    """Plaka tanıma sonucu"""
    plate_number: str
    confidence: float
    is_valid: bool
    coordinates: Tuple[int, int, int, int]
    raw_text: str

class TurkishLicensePlateRecognizer:
    """Türk plaka tanıma sınıfı - optimize edilmiş versiyon"""
    
    # Türk plaka formatları
    PLATE_PATTERNS = [
        r'^\d{2}\s+[A-Z]{1,3}\s+\d{1,4}$',  # 34 AB 1234
        r'^\d{2}\s+[A-Z]{2,3}\s+\d{2,4}$',  # 34 ATA 123
    ]
    
    # Şehir kodları (1-81)
    VALID_CITY_CODES = set(f"{i:02d}" for i in range(1, 82))
    
    def __init__(self, languages: List[str] = ['tr', 'en']):
        """OCR okuyucuyu başlat"""
        self.reader = easyocr.Reader(languages, gpu=True)
        logger.info("EasyOCR başlatıldı")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü ön işleme"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kontrast artırma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gürültü azaltma
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def detect_plate_regions(self, image: np.ndarray) -> List[PlateCandidate]:
        """Plaka bölgelerini tespit et"""
        gray = self.preprocess_image(image)
        candidates = []
        
        # Yöntem 1: Kenar tespiti
        candidates.extend(self._detect_by_edges(image, gray))
        
        # Yöntem 2: Morfolojik işlemler
        candidates.extend(self._detect_by_morphology(image, gray))
        
        # Çakışan bölgeleri filtrele
        filtered = self._filter_overlapping_regions(candidates)
        
        # Eğer hiç plaka bulunamazsa tüm resmi dene
        if not filtered:
            h, w = image.shape[:2]
            filtered = [PlateCandidate(
                region=image,
                coordinates=(0, 0, w, h),
                confidence=0.5,
                area=w * h,
                aspect_ratio=w / h,
                method="full_image"
            )]
        
        return filtered[:3]  # En fazla 3 aday
    
    def _detect_by_edges(self, image: np.ndarray, gray: np.ndarray) -> List[PlateCandidate]:
        """Kenar tespiti ile plaka bölgelerini bul"""
        candidates = []
        
        # Kenar tespiti
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if self._is_valid_plate_size(w, h):
                candidates.append(PlateCandidate(
                    region=image[y:y+h, x:x+w],
                    coordinates=(x, y, w, h),
                    confidence=0.7,
                    area=w * h,
                    aspect_ratio=w / h,
                    method="edge_detection"
                ))
        
        return candidates
    
    def _detect_by_morphology(self, image: np.ndarray, gray: np.ndarray) -> List[PlateCandidate]:
        """Morfolojik işlemler ile plaka bölgelerini bul"""
        candidates = []
        
        # Blackhat işlemi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Gradient
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        grad_x = (grad_x * 255 / grad_x.max()).astype(np.uint8)
        
        # Closing ve threshold
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel_close)
        thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Kontur bulma
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if self._is_valid_plate_size(w, h):
                candidates.append(PlateCandidate(
                    region=image[y:y+h, x:x+w],
                    coordinates=(x, y, w, h),
                    confidence=0.8,
                    area=w * h,
                    aspect_ratio=w / h,
                    method="morphology"
                ))
        
        return candidates
    
    def _is_valid_plate_size(self, width: int, height: int) -> bool:
        """Plaka boyutlarını kontrol et"""
        area = width * height
        aspect_ratio = width / height
        
        return (1000 < area < 100000 and 
                2.5 <= aspect_ratio <= 7.0)
    
    def _filter_overlapping_regions(self, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Çakışan bölgeleri filtrele"""
        if not candidates:
            return []
        
        # Güven skoruna göre sırala
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        filtered = []
        
        for candidate in sorted_candidates:
            x1, y1, w1, h1 = candidate.coordinates
            overlap = False
            
            for existing in filtered:
                x2, y2, w2, h2 = existing.coordinates
                
                # Çakışma kontrolü
                if (x1 < x2 + w2 and x1 + w1 > x2 and 
                    y1 < y2 + h2 and y1 + h1 > y2):
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(candidate)
        
        return filtered
    
    def enhance_plate_image(self, plate_region: np.ndarray) -> np.ndarray:
        """Plaka görüntüsünü OCR için optimize et"""
        # Gri tonlamaya çevir
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # Boyutu artır
        height, width = gray.shape
        scale_factor = max(2, 150 // height)
        resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Adaptive threshold
        processed = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Morfolojik temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_from_plate(self, plate_region: np.ndarray) -> Optional[PlateResult]:
        """Plaka bölgesinden metin çıkar"""
        # Görüntüyü optimize et
        processed = self.enhance_plate_image(plate_region)
        
        # OCR uygula
        ocr_results = self.reader.readtext(processed)
        
        if not ocr_results:
            return None
        
        # Sonuçları birleştir
        text_parts = []
        confidences = []
        
        # Koordinata göre sırala (soldan sağa)
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][0])
        
        for (bbox, text, confidence) in sorted_results:
            # TR etiketini filtrele
            if text.upper().strip() != 'TR' and confidence > 0.3:
                text_parts.append(text.strip())
                confidences.append(confidence)
        
        if not text_parts:
            return None
        
        # Metni birleştir
        combined_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Plaka formatına dönüştür
        formatted_plate = self._format_plate_text(combined_text)
        is_valid = self._validate_plate_format(formatted_plate)
        
        return PlateResult(
            plate_number=formatted_plate,
            confidence=avg_confidence,
            is_valid=is_valid,
            coordinates=(0, 0, 0, 0),  # Bölge içi koordinat
            raw_text=combined_text
        )
    
    def _format_plate_text(self, text: str) -> str:
        """Plaka metnini formatla"""
        # Sadece alfanümerik karakterler
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # OCR hatalarını düzelt
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'G': '6'
        }
        
        # Rakamları ve harfleri ayır
        numbers = ''.join(c for c in clean_text if c.isdigit())
        letters = ''.join(c for c in clean_text if c.isalpha())
        
        # Türk plaka formatına dönüştür
        if len(numbers) >= 3 and len(letters) >= 1:
            # İlk 2 rakam şehir kodu
            if len(numbers) >= 2:
                city_code = numbers[:2]
                remaining_numbers = numbers[2:]
                
                # Formatla: 34 AB 1234
                if remaining_numbers:
                    return f"{city_code} {letters} {remaining_numbers}"
                else:
                    return f"{city_code} {letters}"
        
        # Basit format
        return f"{numbers[:2] if len(numbers) >= 2 else numbers} {letters} {numbers[2:] if len(numbers) > 2 else ''}"
    
    def _validate_plate_format(self, plate_text: str) -> bool:
        """Plaka formatını doğrula"""
        # Fazla boşlukları temizle
        cleaned = re.sub(r'\s+', ' ', plate_text.strip())
        
        # Pattern kontrolü
        for pattern in self.PLATE_PATTERNS:
            if re.match(pattern, cleaned):
                # Şehir kodu kontrolü
                city_code = cleaned[:2]
                if city_code in self.VALID_CITY_CODES:
                    return True
        
        return False
    
    def recognize_plate(self, image_path: str) -> Dict:
        """Ana plaka tanıma fonksiyonu"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Görüntü yüklenemedi"}
            
            logger.info(f"Görüntü yüklendi: {image.shape}")
            
            # Plaka bölgelerini tespit et
            candidates = self.detect_plate_regions(image)
            
            if not candidates:
                return {"error": "Plaka tespit edilemedi"}
            
            logger.info(f"{len(candidates)} plaka adayı bulundu")
            
            # Her aday için OCR
            results = []
            for i, candidate in enumerate(candidates):
                logger.info(f"Plaka {i+1} işleniyor...")
                
                plate_result = self.extract_text_from_plate(candidate.region)
                
                if plate_result:
                    result_dict = {
                        "plate_number": plate_result.plate_number,
                        "confidence": round(plate_result.confidence, 3),
                        "is_valid": plate_result.is_valid,
                        "coordinates": candidate.coordinates,
                        "raw_text": plate_result.raw_text,
                        "method": candidate.method
                    }
                    results.append(result_dict)
                    logger.info(f"Plaka bulundu: {plate_result.plate_number}")
                else:
                    results.append({
                        "error": "Metin okunamadı",
                        "coordinates": candidate.coordinates,
                        "method": candidate.method
                    })
            
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Hata: {e}")
            return {"error": str(e)}
    
    def batch_process_images(self, image_pattern: str = "resim*.png") -> Dict[str, Optional[str]]:
        """Toplu görüntü işleme - sadece plaka sonuçlarını döndür"""
        import glob
        
        image_files = sorted(glob.glob(image_pattern))
        results = {}
        
        for image_file in image_files:
            plate = self.get_best_plate(image_file)
            results[image_file] = plate
            
        return results
    
    def save_results_to_file(self, results: Dict[str, Optional[str]], filename: str = "plaka_sonuclari.txt"):
        """Sonuçları dosyaya kaydet"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("PLAKA TANIMA SONUÇLARI\n")
            f.write("=" * 40 + "\n\n")
            
            for image_file, plate in results.items():
                if plate:
                    f.write(f"{image_file}: {plate}\n")
                else:
                    f.write(f"{image_file}: Plaka bulunamadı\n")
            
            successful = sum(1 for plate in results.values() if plate)
            total = len(results)
            f.write(f"\nBaşarı oranı: {successful}/{total} ({successful/total*100:.1f}%)\n")
        
        print(f"💾 Sonuçlar {filename} dosyasına kaydedildi")

    def get_best_plate(self, image_path: str) -> Optional[str]:
        """En iyi plaka sonucunu döndür"""
        result = self.recognize_plate(image_path)
        
        if "error" in result:
            return None
        
        # En yüksek güven skoruna sahip geçerli plakayı seç
        valid_plates = [r for r in result["results"] 
                       if "plate_number" in r and r.get("is_valid", False)]
        
        if valid_plates:
            best_plate = max(valid_plates, key=lambda x: x["confidence"])
            return best_plate["plate_number"]
        
        # Geçerli plaka yoksa en yüksek güven skoruna sahip olanı döndür
        all_plates = [r for r in result["results"] if "plate_number" in r]
        if all_plates:
            best_plate = max(all_plates, key=lambda x: x["confidence"])
            return best_plate["plate_number"]
        
        return None


def process_multiple_images(recognizer, image_files):
    """Birden fazla görüntüyü işle"""
    all_results = {}
    
    for image_file in image_files:
        print(f"\n📁 İşlenen dosya: {image_file}")
        print("-" * 60)
        
        result = recognizer.recognize_plate(image_file)
        all_results[image_file] = result
        
        if "error" in result:
            print(f"❌ Hata: {result['error']}")
            continue
        
        # En iyi plaka sonucunu göster
        best_plate = recognizer.get_best_plate(image_file)
        if best_plate:
            print(f"🏆 SONUÇ: {best_plate}")
        else:
            print("❌ Plaka bulunamadı")
        
        # Detaylı sonuçları göster
        for i, plate_result in enumerate(result["results"], 1):
            if "plate_number" in plate_result:
                print(f"   📋 Aday {i}: {plate_result['plate_number']} "
                      f"({plate_result['confidence']:.1%} güven, "
                      f"{'geçerli' if plate_result['is_valid'] else 'geçersiz'})")
    
    return all_results

def main():
    """Ana fonksiyon"""
    recognizer = TurkishLicensePlateRecognizer()
    
    # Hızlı toplu işleme seçeneği
    print("🚀 Hızlı toplu işleme için 'h' tuşuna basın, detaylı işleme için Enter'a basın...")
    choice = input().strip().lower()
    
    if choice == 'h':
        # Hızlı toplu işleme
        results = recognizer.batch_process_images("resim*.png")
        
        print("\n" + "="*50)
        print("🚗 HIZLI TOPLU PLAKA TANIMA")
        print("="*50)
        
        for image_file, plate in results.items():
            if plate:
                print(f"✅ {image_file}: {plate}")
            else:
                print(f"❌ {image_file}: Plaka bulunamadı")
        
        successful = sum(1 for plate in results.values() if plate)
        total = len(results)
        print(f"\n📈 BAŞARI ORANI: {successful}/{total} ({successful/total*100:.1f}%)")
        
        # Sonuçları dosyaya kaydet
        recognizer.save_results_to_file(results)
        return
    
    # Önce resim1.png - resim10.png dosyalarını ara
    numbered_files = []
    for i in range(1, 11):
        filename = f"resim{i}.png"
        if os.path.exists(filename):
            numbered_files.append(filename)
    
    # Eğer numaralı dosyalar varsa onları işle
    if numbered_files:
        print("🔍 Numaralı görüntü dosyaları bulundu:")
        for file in numbered_files:
            print(f"   ✅ {file}")
        
        print("\n" + "="*60)
        print("🚗 TOPLU PLAKA TANIMA SONUÇLARI")
        print("="*60)
        
        all_results = process_multiple_images(recognizer, numbered_files)
        
        # Özet sonuçları göster
        print("\n" + "="*60)
        print("📊 ÖZET SONUÇLAR")
        print("="*60)
        
        successful_results = []
        for image_file, result in all_results.items():
            best_plate = recognizer.get_best_plate(image_file)
            if best_plate:
                successful_results.append((image_file, best_plate))
                print(f"✅ {image_file}: {best_plate}")
            else:
                print(f"❌ {image_file}: Plaka bulunamadı")
        
        print(f"\n📈 BAŞARI ORANI: {len(successful_results)}/{len(numbered_files)} "
              f"({len(successful_results)/len(numbered_files)*100:.1f}%)")
        
        return
    
    # Eğer numaralı dosyalar yoksa, diğer görüntü dosyalarını ara
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        import glob
        image_files.extend(glob.glob(ext))
    
    if not image_files:
        print("❌ Hiç görüntü dosyası bulunamadı!")
        print("💡 Lütfen resim1.png, resim2.png, ... resim10.png formatında dosyalar ekleyin")
        return
    
    # Tek dosya işleme (eski davranış)
    image_path = image_files[0]
    print(f"📁 Kullanılan dosya: {image_path}")
    
    # Plaka tanıma
    result = recognizer.recognize_plate(image_path)
    
    print("\n" + "="*50)
    print("🚗 PLAKA TANIMA SONUÇLARI")
    print("="*50)
    
    if "error" in result:
        print(f"❌ Hata: {result['error']}")
        return
    
    # Sonuçları göster
    for i, plate_result in enumerate(result["results"], 1):
        print(f"\n📋 Plaka {i}:")
        print("-" * 25)
        
        if "plate_number" in plate_result:
            print(f"🔢 Plaka: {plate_result['plate_number']}")
            print(f"📊 Güven: {plate_result['confidence']:.1%}")
            print(f"✅ Geçerli: {'Evet' if plate_result['is_valid'] else 'Hayır'}")
            print(f"📝 Ham metin: {plate_result['raw_text']}")
            print(f"🔍 Yöntem: {plate_result['method']}")
        else:
            print(f"❌ {plate_result.get('error', 'Bilinmeyen hata')}")
    
    # En iyi plaka
    best_plate = recognizer.get_best_plate(image_path)
    if best_plate:
        print(f"\n🏆 EN İYİ SONUÇ: {best_plate}")
    else:
        print("\n❌ Geçerli plaka bulunamadı")


if __name__ == "__main__":
    main()
