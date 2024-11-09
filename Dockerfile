FROM python:3.11-slim

RUN apt update -y && apt install awscli -y
WORKDIR /verta-chatbot

COPY . /verta-chatbot
RUN pip install -r requirements.txt

ENV HOST 0.0.0.0

CMD ["python3", "src/serve.py"]