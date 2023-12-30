import os

from flask import jsonify, request, Flask
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
stop_stream = False


def process(query):
    past_key_values, history = None, []
    global stop_stream

    current_length = 0
    res=""
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                temperature=0.01,
                                                                past_key_values=past_key_values,
                                                                return_past_key_values=True):
        if stop_stream:
            stop_stream = False
            break
        else:
            print(response[current_length:], end="", flush=True)
            res+=response[current_length:]
            current_length = len(response)
    return res

@app.route('/ask', methods=['GET'])
def do_ask():
    query = request.args.get('query', default='', type=str)
    return jsonify({'code': 200, 'ask': query, 'answer': process(query)})

if __name__ == "__main__":
    app.run()