# ESG Scoring Report Dashboard

Ứng dụng Streamlit cho phép trực quan hóa và phân tích kết quả ESG từ tài liệu PDF.

## Tính năng

- Upload tệp PDF để phân tích dựa trên tiêu chí ESG
- Điều chỉnh các tham số phân tích (threshold, top_k)
- Hiển thị kết quả dưới dạng các tab E-S-G
- Trực quan hóa dữ liệu bằng biểu đồ và bảng
- Tùy chọn sử dụng API hoặc xử lý cục bộ
- Xuất kết quả sang JSON

## Yêu cầu

Cài đặt các thư viện cần thiết:

```bash
pip install -r UI/requirements-ui.txt
```

## Chạy ứng dụng

Từ thư mục gốc của dự án, chạy lệnh:

```bash
streamlit run UI/demo_streamlit.py
```

Ứng dụng sẽ chạy trên cổng mặc định 8501 và mở trình duyệt web tự động. Bạn cũng có thể truy cập http://localhost:8501 trong trình duyệt của mình.

## Sử dụng API

Mặc định, ứng dụng sẽ sử dụng API để xử lý phân tích ESG. Đảm bảo rằng FastAPI backend đã được khởi động trước khi sử dụng:

```bash
uvicorn src.app:app --reload
```

Hoặc bạn có thể tắt tùy chọn "Use API endpoint" để xử lý cục bộ.
