# src/ocr_engine.py (Dùng PyTesseract)

import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import numbers # Cần để kiểm tra số trong visualize

# --- Cài đặt Tesseract (Chỉ cần thiết nếu không thêm vào PATH) ---
# Ví dụ:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tắt draw_ocr cũ
draw_ocr = None 
print("Cảnh báo: Đã chuyển sang PyTesseract. Đảm bảo Tesseract Engine đã được cài đặt và thêm vào PATH.")


# ======================================================
# 1. HÀM RUN OCR (PyTesseract)
# ======================================================
# Đổi tên hàm thành run_tesseract_ocr để rõ ràng
def run_tesseract_ocr(image_path):
    # Đọc ảnh bằng OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    img_h, img_w = img_bgr.shape[:2]

    # Chuyển ảnh BGR (OpenCV) sang RGB (Tesseract)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    words = []
    boxes = [] 
    
    try:
        # Chạy Tesseract với ngôn ngữ Tiếng Việt (vie)
        data = pytesseract.image_to_data(
            img_rgb, 
            lang='vie', 
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError("Tesseract Engine không được tìm thấy. Vui lòng cài đặt Tesseract và kiểm tra PATH.") from e
    except Exception as e:
        print(f"DEBUG: Lỗi khi gọi Tesseract: {e}")
        return words, boxes, (img_w, img_h)

    # Phân tích kết quả chi tiết từ Tesseract
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        conf = data['conf'][i]
        text = str(data['text'][i]).strip()
        
        # Chỉ lấy các từ có độ tin cậy trên 70 và không rỗng
        # Tesseract trả về conf là string nếu không nhận diện được
        if text and text != " " and conf != "-1":
            conf_int = int(conf)
            
            if conf_int > 70:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                
                # Tesseract trả về hộp chữ nhật [x_min, y_min, x_max, y_max]
                # Chuyển về định dạng 8 tọa độ (quad) cho phù hợp với code LayoutLM
                # [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                quad_box = [x, y, x + w, y, x + w, y + h, x, y + h]
                
                words.append(text)
                boxes.append(quad_box)
            
    print(f"DEBUG: Tesseract extracted {len(words)} words")
     # In ra các từ đã trích xuất
    if words:
        print("=" * 50)
        print("CÁC TỪ TRÍCH XUẤT ĐƯỢC TỪ OCR:")
        print("=" * 50)
        for idx, word in enumerate(words, 1):
            print(f"{idx:3d}. {word}")
        print("=" * 50)
    else:
        print("CẢNH BÁO: Không trích xuất được từ nào từ ảnh!")
    return words, boxes, (img_w, img_h)



# ======================================================
# 2. Visualization (Vẽ bằng PIL - Giữ nguyên logic cũ)
# ======================================================
def visualize_ocr(image_path, boxes, texts):
    """Vẽ đa giác từ list 8 tọa độ; bỏ qua box sai định dạng."""
    img = Image.open(image_path).convert("RGB")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    if not boxes or not isinstance(boxes, list):
        return img

    for box in boxes:
        try:
            if not (isinstance(box, (list, tuple)) and len(box) == 8 and all(isinstance(v, numbers.Number) for v in box)):
                continue
            # Chuyển từ [x1,y1,x2,y2,x3,y3,x4,y4] về [[x1,y1], ...]
            poly = [[box[i], box[i + 1]] for i in range(0, 8, 2)]
            draw.polygon([tuple(p) for p in poly], outline="red", width=2)
        except Exception as e:
            print(f"DEBUG: visualize lỗi box={box}: {e}")
            continue

    return img

if __name__ == "__main__":
    pass