FROM python:3.12-slim

RUN apt update -y && apt install awscli -y

WORKDIR /verta-chatbot

RUN pip install poetry

COPY . /verta-chatbot

RUN poetry config virtualenvs.create false 
RUN poetry install --no-dev --no-root --verbose

ENV HOST 0.0.0.0
ENV PORT 80

CMD ["poetry", "python", "serve.py"]