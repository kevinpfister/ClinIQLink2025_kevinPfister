# Dockerfile
FROM python:3.11

# (Optional) Set environment variables for Huggingface
ENV HF_HOME=D:/.cache/huggingface
ENV TRANSFORMERS_CACHE=D:/.cache/huggingface/hub

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/rag.py"]