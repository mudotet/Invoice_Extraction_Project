# app.py (Nằm ngoài thư mục src)

import streamlit as st
from PIL import Image
import os
import io
import time
import uuid

# --- THAY ĐỔI: Dùng Import Tuyệt đối ---
from src.inference import predict_kie
from src.ocr_engine import visualize_ocr 
# from src.utils import unnormalize_box # Không cần unnormalize_box trong app.py

# --- Cấu hình Streamlit ---
st.set_page_config(
    page_title="Vietnamese Receipt KIE App",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧾 Trích xuất Dữ liệu Hóa đơn Tiếng Việt (LayoutLM + Tesseract/pytesseract)")
st.write("Ứng dụng Data Mining/KIE sử dụng LayoutLM và Tesseract (pytesseract) để nhận dạng Người bán, Ngày và Tổng tiền.")

# Tạo thư mục tạm thời để lưu ảnh
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Chức năng chính ---

uploaded_file = st.file_uploader("Tải lên hình ảnh hóa đơn (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Lưu file tạm thời
    file_bytes = uploaded_file.read()
    temp_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_{uploaded_file.name}")
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Hóa đơn")
        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Kết quả Trích xuất")
        
        if st.button("Trích xuất Thông tin", key="extract_btn"):
            with st.spinner('Đang chạy OCR và dự đoán LayoutLM...'):
                start_time = time.time()
                
                # 2. Chạy pipeline dự đoán
                results = predict_kie(temp_filename)
                
                end_time = time.time()
                
                if isinstance(results, dict) and 'error' in results:
                    st.error(results['error'])
                else:
                    final_results, extracted_details = results
                    
                    st.success(f"Trích xuất hoàn tất trong {end_time - start_time:.2f} giây!")
                    
                    # 3. Hiển thị kết quả gộp
                    st.json(final_results)
                    
                    # 4. (Tùy chọn) Hiển thị ảnh kèm Bounding Box
                    st.markdown("---")
                    st.subheader("Trực quan hóa OCR")
                    
                    # Lấy dữ liệu từ extracted_details
                    viz_words = []
                    viz_boxes = []
                    for key, items in extracted_details.items():
                        for item in items:
                            viz_words.append(item['text'])
                            viz_boxes.append(item['box'])
                            
                    # Vẽ bằng hàm visualize_ocr từ src/ocr_engine.py
                    try:
                        annotated_image = visualize_ocr(temp_filename, viz_boxes, viz_words)
                        st.image(annotated_image, caption="Các trường quan trọng được đánh dấu", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Không thể trực quan hóa: {e}")

    # 5. Dọn dẹp file tạm
    if os.path.exists(temp_filename):
        os.remove(temp_filename)