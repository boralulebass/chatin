import speech_recognition as sr
import json
from vosk import Model, KaldiRecognizer

def transcribe_audio():
    # Türkçe dil modeli ve model dosyasının yolu
    MODEL_PATH = r"C:\Users\nural\PycharmProjects\Chat-In\testfiles\vosk-model-small-tr-0.3"

    # Vosk modelini yükleme
    model = Model(MODEL_PATH)

    # SpeechRecognition kütüphanesini kullanarak mikrofonu başlatma
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Sizi Dinliyorum...")

    try:
        with microphone as source:
            #recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)  # 10 saniyeye kadar ses kaydedin

        print("Sesiniz kaydedildi, hemen döneceğim...")

        # Ses verisini metne çevirme
        audio_data = audio.get_wav_data(convert_rate=16000, convert_width=2)  # Ses verisini WAV formatına dönüştürme ve Vosk modeline uygun formata getirme
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(audio_data)
        result_str = recognizer.Result()
        result_dict = json.loads(result_str)
        text = result_dict["text"]

        return text

    except Exception as e:
        return "Bir hata oluştu: {0}".format(e)

if __name__ == "__main__":
    result_text = transcribe_audio()
    print("Çeviri:", result_text)



