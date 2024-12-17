from caption_generator import predict
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=['POST'])
def index():
    data = request.get_data()
    caption = predict(data)
    return {'caption': caption}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
