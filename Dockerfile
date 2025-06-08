# Sử dụng image Python chính thức, nhẹ
FROM python:3.10-slim

# Thiết lập biến môi trường (không hardcode API key!)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Tạo thư mục ứng dụng
WORKDIR /app

# Copy requirements trước (để cache được tốt hơn)
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Chạy ứng dụng FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
