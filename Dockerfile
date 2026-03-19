FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV PORT=8080
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
