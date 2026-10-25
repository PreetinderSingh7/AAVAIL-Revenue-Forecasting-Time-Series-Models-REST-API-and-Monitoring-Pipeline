FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# ingest (synthetic demo data), run EDA, train and compare models at build time
RUN python ingest.py && python eda.py && python model.py
EXPOSE 8080
CMD ["python", "app.py"]
