import os
import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
from src.ocr_engine import run_tesseract_ocr
from src.utils import convert_quad_to_box, normalize_box

# --- Cấu hình ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "layoutlm_best")

try:
    print(f"Đang tải model từ: {MODEL_PATH}")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
    ID_TO_LABEL = model.config.id2label
    model.eval()
    print(f"✓ Model LayoutLM đã được tải thành công với {len(ID_TO_LABEL)} nhãn.")
    print(f"Nhãn: {ID_TO_LABEL}")
except Exception as e:
    print(f"✗ Lỗi khi tải LayoutLM Model: {e}")
    print(f"Đường dẫn kiểm tra: {MODEL_PATH}")
    tokenizer = None
    model = None    
    ID_TO_LABEL = {}

def predict_kie(image_path):
    if model is None or tokenizer is None:
        return {"error": "Model/Tokenizer chưa được tải thành công. Kiểm tra MODEL_PATH."}

    print(f"DEBUG: Checking image at: {image_path}")
    print(f"DEBUG: File exists: {os.path.exists(image_path)}")

    try:
        words_raw, boxes_quad_raw, (width, height) = run_tesseract_ocr(image_path)
        print(f"DEBUG: OCR extracted {len(words_raw)} words")
    except Exception as e:
        return {"error": f"Lỗi trong quá trình OCR: {e}"}

    if not words_raw:
        return {"error": "Không trích xuất được văn bản nào từ hóa đơn."}

    # Đồng bộ độ dài
    if len(boxes_quad_raw) != len(words_raw):
        min_len = min(len(boxes_quad_raw), len(words_raw))
        print(f"DEBUG: Mismatch words({len(words_raw)}) vs boxes({len(boxes_quad_raw)}). Cắt về {min_len}.")
        words_raw = words_raw[:min_len]
        boxes_quad_raw = boxes_quad_raw[:min_len]

    # Chuẩn hóa bbox -> 0..1000
    boxes_norm = []
    for idx, quad in enumerate(boxes_quad_raw):
        try:
            box_rect = convert_quad_to_box(quad)
            box_norm = normalize_box(box_rect, width, height)
            boxes_norm.append(box_norm)
        except Exception as e:
            print(f"DEBUG: Lỗi convert bbox ở index {idx}: {e}. Gán [0,0,0,0].")
            boxes_norm.append([0, 0, 0, 0])

    # Debug phạm vi bbox
    if boxes_norm:
        mins = [min(b) for b in boxes_norm]
        maxs = [max(b) for b in boxes_norm]
        print(f"DEBUG: BBox range min={min(mins)} max={max(maxs)} (expected 0..1000)")
        # In 5 bbox đầu
        print(f"DEBUG: First 5 bboxes (norm): {boxes_norm[:5]}")

    max_len = 512
    encoding = tokenizer(
        words_raw,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

    # Ánh xạ bbox theo token (wordpiece)
    word_ids = encoding.word_ids()
    token_sequence_length = encoding.input_ids.shape[1]

    padded_boxes_norm = []
    for token_idx in range(token_sequence_length):
        word_idx = word_ids[token_idx]
        if word_idx is not None and word_idx < len(boxes_norm):
            padded_boxes_norm.append(boxes_norm[word_idx])
        else:
            padded_boxes_norm.append([0, 0, 0, 0])

    # Ép int + clamp cứng 0..1000
    for i in range(len(padded_boxes_norm)):
        b = padded_boxes_norm[i]
        b0 = max(0, min(1000, int(b[0])))
        b1 = max(0, min(1000, int(b[1])))
        b2 = max(0, min(1000, int(b[2])))
        b3 = max(0, min(1000, int(b[3])))
        # Đảm bảo đúng thứ tự (x0<=x1, y0<=y1)
        if b2 < b0: b0, b2 = b2, b0
        if b3 < b1: b1, b3 = b3, b1
        padded_boxes_norm[i] = [b0, b1, b2, b3]

    bbox_tensor = torch.tensor([padded_boxes_norm], dtype=torch.long)
    # Clamp lần cuối (phòng ngừa)
    bbox_tensor = torch.clamp(bbox_tensor, min=0, max=1000)

    # Kiểm tra min/max lần cuối
    minv = int(bbox_tensor.min().item())
    maxv = int(bbox_tensor.max().item())
    print(f"DEBUG: Final bbox tensor min={minv} max={maxv}")

    encoding['bbox'] = bbox_tensor

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                bbox=encoding.bbox
            )
    except IndexError as e:
        # Gói lỗi cho UI + log chi tiết
        print("DEBUG: IndexError khi chạy model. In 10 bbox đầu:")
        print(padded_boxes_norm[:10])
        return {"error": f"Lỗi bbox ngoài 0..1000. min={minv}, max={maxv}. Vui lòng kiểm tra normalize_box/width/height."}

    predictions = outputs.logits.argmax(dim=2).squeeze().tolist()

    extracted_tokens = []
    for token_idx, pred_id in enumerate(predictions):
        pred_label_raw = ID_TO_LABEL.get(pred_id, "O")
        word_idx = word_ids[token_idx]
        if word_idx is not None and pred_label_raw != "O":
            tag_type = pred_label_raw.split("-")[-1]
            if not extracted_tokens or extracted_tokens[-1]['word_idx'] != word_idx:
                extracted_tokens.append({
                    "word_idx": word_idx,
                    "label": tag_type,
                    "text": words_raw[word_idx],
                    "box": boxes_quad_raw[word_idx]
                })

    final_extracted_info = {"SELLER": [], "TIMESTAMP": [], "TOTAL_COST": []}
    for item in extracted_tokens:
        if item['label'] in final_extracted_info:
            final_extracted_info[item['label']].append(item)

    final_results = {}
    for key, entity_list in final_extracted_info.items():
        final_results[f"{key} (Gộp)"] = " ".join([item['text'] for item in entity_list]) if entity_list else ""

    return final_results, final_extracted_info

if __name__ == "__main__":
    # Ví dụ cách sử dụng khi file này chạy độc lập
    # Lưu ý: Cần có file ảnh test.png và model đã train đúng
    TEST_IMAGE = "test.png" 
    
    # Giả lập hàm utils để code chạy được nếu thiếu file utils
    if 'convert_quad_to_box' not in locals():
        print("⚠ Using mock convert_quad_to_box.")
        def convert_quad_to_box(quad): return [min(quad[::2]), min(quad[1::2]), max(quad[::2]), max(quad[1::2])]
        def normalize_box(box, w, h): return [int(box[0]*1000/w), int(box[1]*1000/h), int(box[2]*1000/w), int(box[3]*1000/h)]

    # Thêm logic test (Giả định rằng bạn đã có model và ảnh test)
    if os.path.exists(TEST_IMAGE) and model is not None:
        print("\n--- Bắt đầu Inference ---")
        merged, detailed = predict_kie(TEST_IMAGE)
        
        print("\n--- KẾT QUẢ GỘP ---")
        for k, v in merged.items():
            print(f"{k}: {v}")
            
        print("\n--- CHI TIẾT TỪ (Dùng cho Visualization) ---")
        for label, items in detailed.items():
            for item in items:
                print(f"[{item['label']}] Text: {item['text']}, Box: {item['box']}")
    else:
        print("\n--- LƯU Ý ---")
        print(f"Cần file ảnh '{TEST_IMAGE}' và model đã train tại '{MODEL_PATH}' để chạy test.")