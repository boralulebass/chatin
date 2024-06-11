import random
import json
import torch
import tkinter as tk
from tkinter import scrolledtext
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from whispers import transcribe_audio

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

root = tk.Tk()
root.title("Chat-In")
root.geometry("500x600")

window_width = 500
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

header = tk.Label(root, text="Chat-In", font=("Times New Roman", 24), bg="#ffffff", fg="#333333")
header.pack(pady=10)

chat_display = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD, bg="#f4f4f4", fg="#333333",
                                         font=("Times New Roman", 12))
chat_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

message_entry = tk.Entry(root, font=("Times New Roman", 14), bg="#e0e0e0")
message_entry.pack(pady=10, padx=10, fill=tk.X)


def send_message(event=None):
    user_message = message_entry.get()
    if user_message:
        chat_display.configure(state='normal')
        chat_display.insert(tk.END, "Siz: " + user_message)
        chat_display.configure(state='disabled')
        message_entry.delete(0, tk.END)
        root.after(500, lambda: show_typing_animation(user_message, 0))



def show_typing_animation(user_message, dot_count):
    if dot_count < 3:
        chat_display.configure(state='normal')
        chat_display.insert(tk.END, '.')
        chat_display.configure(state='disabled')
        chat_display.yview(tk.END)
        root.after(500, lambda: show_typing_animation(user_message, dot_count + 1))
    else:
        handle_response(user_message)


def handle_response(sentence):
    chat_display.configure(state='normal')
    chat_display.delete('end-4c', tk.END)

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
                chat_display.insert(tk.END, f"\n{bot_name}: {bot_response}\n")
                chat_display.yview(tk.END)
                break
    elif 0.95 > prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                chat_display.insert(tk.END, f"\n{bot_name}: Emin değilim ama aradığınız cevap bu olabilir mi? :{bot_response}\n")
                chat_display.yview(tk.END)
                break
    else:
        chat_display.insert(tk.END, f"\n{bot_name}: Elimdeki dokümanlar ile buna cevap veremiyorum... \n")
        chat_display.yview(tk.END)

    chat_display.configure(state='disabled')



def listen_audio():
    sentence = transcribe_audio()
    chat_display.configure(state='normal')
    chat_display.insert(tk.END, "Siz: " + sentence)
    chat_display.configure(state='disabled')
    root.after(500, lambda: show_typing_animation(sentence, 0))





message_entry.bind("<Return>", send_message)

send_button = tk.Button(root, text="Gönder", font=("Helvetica", 14), bg="#4caf50", fg="#ffffff", command=send_message)
send_button.pack(pady=10)

mic_button = tk.Button(root, text="🎤", font=("Helvetica", 14), bg="#2196f3", fg="#ffffff", command=listen_audio)
mic_button.pack(pady=10)


root.mainloop()
