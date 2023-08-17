FROM python:3.10.12

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "api.py", "--model", "./built_models/gpt.onnx", "--tokenizer", "./tokenizer/tokenizer.pkl"]