#!/usr/bin/env python3
"""
Türk Plaka Tespit Sistemi - YOLOv8 ve EasyOCR ile
Turkish License Plate Detection System using YOLOv8 and EasyOCR
"""

import cv2
import numpy as np
import re
from collections import deque, Counter
from ultralytics import YOLO
import easyocr
import os
import time

# Gerekli modülleri yükle ve model dosyalarını kontrol et
def setup_models():
    """Model dosyalarını yükle ve kontrol et"""
    print("🚀 Model yükleniyor...")
    
    # YOLOv8 modelleri
    try:
        vehicle_model = YOLO('yolov8n.pt')  # Araç tespiti için
        print("✅ Araç tespit modeli yüklendi: yolov8n.pt")
    except Exception as e:
        print(f"❌ Araç modeli yüklenemedi: {e}")
        exit(1)
    
    try:
        plate_model = YOLO('best.pt')  # Plaka tespiti için
        print("✅ Plaka tespit modeli yüklendi: best.pt")
    except Exception as e:
        print(f"❌ Plaka modeli yüklenemedi: {e}")
        print("⚠️  best.pt dosyasını proje dizinine yerleştirin")
        exit(1)
    
    # EasyOCR okuyucu (Türkçe ve İngilizce)
    try:
        ocr_reader = easyocr.Reader(['tr', 'en'], gpu=True)
        print("✅ EasyOCR yüklendi (Türkçe ve İngilizce)")
    except Exception as e:
        print(f"❌ EasyOCR yüklenemedi: {e}")
        exit(1)
    
    return vehicle_model, plate_model, ocr_reader

# Global değişkenler
vehicle_model, plate_model, ocr_reader = setup_models()

# Son 20 plaka tespitini saklamak için deque
recent_plates = deque(maxlen=20)

# Türk plaka formatı regex deseni
# Format: 34ABC123 (2 rakam + 1-3 harf + 2-4 rakam)
TURKISH_PLATE_PATTERN = re.compile(r'^[0-9]{2}[A-ZÇĞİÖŞÜ]{1,3}[0-9]{2,4}$')

# Araç sınıfları (COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

def detect_vehicles(frame):
    """
    Çerçevede araçları tespit et
    
    Args:
        frame: Giriş görüntüsü
    
    Returns:
        list: Tespit edilen araçların koordinatları [(x1,y1,x2,y2,confidence), ...]
    """
    results = vehicle_model(frame, verbose=False)
    vehicles = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Sadece araç sınıfları ve yüksek güven skorlu tespitler
                if cls in VEHICLE_CLASSES and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    vehicles.append((x1, y1, x2, y2, conf))
    
    return vehicles

def detect_plates_in_vehicle(vehicle_crop):
    """
    Araç bölgesinde plaka tespit et
    
    Args:
        vehicle_crop: Kırpılmış araç görüntüsü
    
    Returns:
        list: Tespit edilen plakaların koordinatları [(x1,y1,x2,y2,confidence), ...]
    """
    results = plate_model(vehicle_crop, verbose=False)
    plates = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                
                # Plaka için daha düşük eşik değeri
                if conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plates.append((x1, y1, x2, y2, conf))
    
    return plates

def preprocess_plate_image(plate_crop):
    """
    Plaka görüntüsünü OCR için ön işleme
    
    Args:
        plate_crop: Kırpılmış plaka görüntüsü
    
    Returns:
        numpy.ndarray: İşlenmiş görüntü
    """
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme
    equalized = cv2.equalizeHist(gray)
    
    # OTSU eşikleme
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfolojik işlemler ile temizleme
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_plate_text(plate_crop):
    """
    Plaka görüntüsünden metni çıkar
    
    Args:
        plate_crop: Kırpılmış plaka görüntüsü
    
    Returns:
        str or None: Geçerli plaka metni veya None
    """
    try:
        # Görüntüyü ön işleme
        processed = preprocess_plate_image(plate_crop)
        
        # EasyOCR ile metin çıkarma
        results = ocr_reader.readtext(processed)
        
        for (bbox, text, confidence) in results:
            if confidence > 0.6:  # Minimum güven eşiği
                # Metni temizle (sadece harf ve rakam)
                clean_text = re.sub(r'[^A-ZÇĞİÖŞÜ0-9]', '', text.upper())
                
                # Türk plaka formatını kontrol et
                if TURKISH_PLATE_PATTERN.match(clean_text):
                    return clean_text
        
        return None
    except Exception as e:
        print(f"❌ OCR hatası: {e}")
        return None

def get_most_frequent_plate():
    """
    En sık tespit edilen plakayı döndür
    
    Returns:
        str or None: En sık geçen plaka
    """
    if not recent_plates:
        return None
    
    counter = Counter(recent_plates)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def draw_detection_results(frame, vehicles, vehicle_plates):
    """
    Tespit sonuçlarını çerçeve üzerine çiz
    
    Args:
        frame: Orijinal çerçeve
        vehicles: Tespit edilen araçlar
        vehicle_plates: Araç-plaka eşleşmeleri
    
    Returns:
        numpy.ndarray: Çizimli çerçeve
    """
    result_frame = frame.copy()
    
    # Araçları ve plakaları çiz
    for i, (x1, y1, x2, y2, conf) in enumerate(vehicles):
        # Araç çerçevesi
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Araç etiketi
        vehicle_label = f"Araç: {conf:.2f}"
        cv2.putText(result_frame, vehicle_label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Plaka varsa çiz
        if i in vehicle_plates:
            plate_text = vehicle_plates[i]
            
            # Plaka metni için arka plan
            text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(result_frame, (x1, y2+5), 
                         (x1+text_size[0]+10, y2+text_size[1]+15), (0, 0, 255), -1)
            
            # Plaka metni
            cv2.putText(result_frame, plate_text, (x1+5, y2+text_size[1]+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # En sık geçen plakayı ekranın üstüne yaz
    most_frequent = get_most_frequent_plate()
    if most_frequent:
        top_text = f"EN SIK TESPIT: {most_frequent}"
        text_size = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
        # Arka plan
        cv2.rectangle(result_frame, (10, 10), 
                     (text_size[0]+20, text_size[1]+20), (0, 0, 255), -1)
        
        # Metin
        cv2.putText(result_frame, top_text, (15, text_size[1]+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    return result_frame

def process_video(input_path='input.mp4', output_path='output.mp4'):
    """
    Video dosyasını işle
    
    Args:
        input_path: Giriş video dosyası
        output_path: Çıkış video dosyası
    """
    # Video dosyasının varlığını kontrol et
    if not os.path.exists(input_path):
        print(f"❌ Video dosyası bulunamadı: {input_path}")
        return
    
    # Video yakalama
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {input_path}")
        return
    
    # Video özellikleri
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video bilgileri: {width}x{height}, {fps} FPS, {total_frames} kare")
    
    # Video yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print("🎬 Video işleme başlatılıyor...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Araçları tespit et
            vehicles = detect_vehicles(frame)
            vehicle_plates = {}
            
            # Her araç için plaka tespiti
            for i, (x1, y1, x2, y2, conf) in enumerate(vehicles):
                # Araç bölgesini kırp
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size > 0:
                    # Plaka tespit et
                    plates = detect_plates_in_vehicle(vehicle_crop)
                    
                    # Her plaka için OCR
                    for px1, py1, px2, py2, pconf in plates:
                        # Plaka koordinatlarını orijinal çerçeveye çevir
                        abs_px1, abs_py1 = x1 + px1, y1 + py1
                        abs_px2, abs_py2 = x1 + px2, y1 + py2
                        
                        # Plaka bölgesini kırp
                        plate_crop = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                        
                        if plate_crop.size > 0:
                            # Plaka metnini çıkar
                            plate_text = extract_plate_text(plate_crop)
                            
                            if plate_text:
                                vehicle_plates[i] = plate_text
                                recent_plates.append(plate_text)
                                
                                # Konsola yazdır
                                print(f"✅ Plaka tespit edildi: {plate_text}")
                                
                                break  # İlk geçerli plakayı kullan
            
            # Sonuçları çiz
            result_frame = draw_detection_results(frame, vehicles, vehicle_plates)
            
            # Çıkış videosuna yaz
            out.write(result_frame)
            
            # Görüntüyü göster (isteğe bağlı)
            cv2.imshow('Türk Plaka Tespit Sistemi', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # İlerleme göster
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 İlerleme: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\n⏹️ İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"❌ İşlem hatası: {e}")
    
    finally:
        # Kaynakları serbest bırak
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ İşlem tamamlandı!")
        print(f"⏱️ İşlem süresi: {processing_time:.2f} saniye")
        print(f"🎯 İşlenen kare sayısı: {frame_count}")
        print(f"📁 Çıkış dosyası: {output_path}")
        
        # En sık tespit edilen plakayı göster
        most_frequent = get_most_frequent_plate()
        if most_frequent:
            print(f"🏆 En sık tespit edilen plaka: {most_frequent}")

def main():
    """Ana fonksiyon"""
    print("🇹🇷 Türk Plaka Tespit Sistemi")
    print("=" * 50)
    
    # Giriş dosyasını kontrol et
    input_file = 'input.mp4'
    if not os.path.exists(input_file):
        print(f"⚠️  {input_file} dosyası bulunamadı!")
        print("📁 Lütfen input.mp4 dosyasını proje dizinine yerleştirin")
        return
    
    # Video işleme
    process_video('input.mp4', 'output.mp4')
    
    print("\n🎉 Program başarıyla tamamlandı!")

if __name__ == "__main__":
    main()