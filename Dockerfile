FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY index.html ./index.html

WORKDIR /app/scripts

EXPOSE 7860

CMD ["python", "module_12_api.py", "--host", "0.0.0.0", "--port", "7860"]