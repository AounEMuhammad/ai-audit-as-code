
FROM python:3.11-slim
WORKDIR /app
COPY . /app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit","run","dashboards/app.py","--server.address=0.0.0.0","--server.port=8501"]
