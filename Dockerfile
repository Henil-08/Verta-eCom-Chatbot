FROM python:3.12-slim

RUN apt update -y && apt install awscli -y

WORKDIR /verta-chatbot

COPY . /verta-chatbot
RUN pip install -r requirements.txt

ENV HOST 0.0.0.0
ENV PORT 80

CMD ["python3", "serve.py"]