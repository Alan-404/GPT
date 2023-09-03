from flask import Flask, request, render_template
import onnxruntime as ort
from preprocessing.data import Tokenizer

tokenizer = Tokenizer("./tokenizer/v1/dictionary.pkl")

session = ort.InferenceSession("./built_models/v1/gpt.onnx", providers=['CUDAExecutionProvider', "CPUExecutionProvider"])

app = Flask(__name__)

@app.route("/chat", methods=['POST'])
def index():
    if request.method == 'POST':
        message = request.get_json().get('message')

        digits = tokenizer.text_to_sequences([message], start_token=True, sep_token=True)

        print(digits)

        
        return {'response': "OK"}


if __name__ == '__main__':
    from waitress import serve
    HOST = '127.0.0.1'
    PORT = 8080

    print(f"Server is running at {HOST} with port {PORT}")
    serve(app, host=HOST, port=PORT)

