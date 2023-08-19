FROM python:3.10.12

WORKDIR /app

COPY requirements.txt /app/requirements.txt

COPY preprocess /app/preprocess

COPY /built_models/gpt.onnx /app/built_models/gpt.onnx

COPY api.py /app/api.py

RUN pip install -r /app/requirements.txt

EXPOSE 8000

CMD ["python3", "api.py", "--model", "./built_models/gpt.onnx", "--tokenizer", "./tokenizer/tokenizer.pkl", "--device", "cuda"]