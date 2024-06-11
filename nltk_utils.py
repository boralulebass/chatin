import numpy as np
import nltk
nltk.download('punkt')
from TurkishStemmer import TurkishStemmer
stemmer = TurkishStemmer()


def tokenize(sentence):
    """
    Cümleyi kelimelere yani tokenlerine ayırarak bir array listesi oluşturur
    bir token kelime veya bir sayı/rakam olabilir
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = kelimenin kökünü bulma
    örnek:
    words = ["kaçınmak", "kaçmak", "kaçırmak"]
    words = [stem(w) for w in words]
    -> ["kaç", "kaç", "kaç"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    geriye bag of words arrayini gönder:
    Cümle içindeki her bilinen kelime için 1, bilinmiyorsa 0
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # her kelimeyi köküne ayır
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag