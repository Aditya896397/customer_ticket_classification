FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
EXPOSE 8501
CMD ["/bin/bash","-c","uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
