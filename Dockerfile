FROM python:3.11-slim
#https://github.com/chroma-core/chroma/issues/642
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
ENV PYTHONUNBUFFERED True

WORKDIR /app
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "ChatPDF.py", "--server.port=8080", "--server.address=0.0.0.0"]
