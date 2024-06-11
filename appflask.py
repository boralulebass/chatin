from flask import Flask, request, jsonify, render_template
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from whispers import transcribe_audio

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ChatIn"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message")
    response = handle_response(user_message)
    return jsonify(response)

@app.route('/listen_audio', methods=['POST'])
def listen_audio():
    sentence = transcribe_audio()
    response = handle_response(sentence)
    return jsonify({"message": sentence, "response": response})

def handle_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() >= 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                return {"bot_name": bot_name, "response": bot_response}
    elif 0.95 > prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                return {"bot_name": bot_name, "response": f"Emin değilim ama aradığınız cevap bu olabilir mi? : {bot_response}"}
    else:
        return {"bot_name": bot_name, "response": "Elimdeki dokümanlar ile buna cevap veremiyorum..."}

if __name__ == "__main__":
    app.run(debug=True)
