FROM python:3.10-slim

WORKDIR /app

COPY requirements-frontend.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --default-timeout=200 -r requirements-frontend.txt \
    --retries 10 --timeout 200 \
    -i https://pypi.org/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
