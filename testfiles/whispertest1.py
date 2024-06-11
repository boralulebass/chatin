import whisper
import speech_recognition as sr
import numpy as np
import torch
import warnings
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import pipeline


# FP16 uyarılarını bastırma
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*", category=UserWarning)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Konuşabilirsiniz")
    audio = r.listen(source)
    audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)  # AudioData'yi NumPy dizisine dönüştür

    # Önce veriyi normalize edin
    normalized_audio = audio_data / np.iinfo(np.int16).max

    # Veriyi uygun tensöre dönüştürün
    audio_tensor = torch.tensor(normalized_audio, dtype=torch.float32)

    model = whisper.load_model("large")
    result = model.transcribe(audio_tensor)
    print(result["text"])




import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r',encoding='utf-8') as json_data:
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
print("Selamlar! Konuşmadan çıkmak için 'çık' yazabilirsin..")
while True:
    sentence = input("Sen: ")
    if sentence == "çık":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Anlayamadım...")