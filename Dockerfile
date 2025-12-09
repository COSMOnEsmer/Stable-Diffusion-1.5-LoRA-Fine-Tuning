FROM python:3.10-slim
WORKDIR /app
COPY  requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "mac_silicon_fine_tuned.py"]