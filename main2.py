import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import re
import pickle
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.saving import load_model
from keras.src.utils import pad_sequences
from langdetect import detect
from translate import Translator


# TensorFlow Lite modelini yükleyin
model = tf.keras.models.load_model("model.h5")

# Tokenizer'ı yükleyin
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)


def detect_language_and_translate(text, target_language="en"):
    try:
        detected_language = detect(text)
        print(f"Tespit edilen dil: {detected_language}")

        if detected_language == target_language:
            return text
        translator = Translator(from_lang=detected_language, to_lang=target_language)
        translation = translator.translate(text)
        print(translation)

        return translation
    except Exception as e:
        return f"Hata: {e}"


# TEXT TEMİZLEME
def clean_text(text):
    translate = detect_language_and_translate(text)
    translate = translate.lower()
    translate = re.sub(r'[^a-zA-Z0-9\s]', '', translate)
    return translate


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text_input = data['input']
        cleaned_input = clean_text(text_input)

        # Tokenize ve pad işlemi
        text_seq = tokenizer.texts_to_sequences([cleaned_input])
        text_pad = pad_sequences(text_seq, maxlen=100, padding='post')

        # TensorFlow modeli kullanarak tahmin yapma
        output_data = model.predict(text_pad)
        predicted_label = np.argmax(output_data)

        # Duygu durumları
        duygu_durumlari = {
            0: "sad",
            1: "happy",
            2: "love",
            3: "angry",
            4: "fear",
            5: "surprise"
        }

        duygu = duygu_durumlari.get(predicted_label, "Bilinmeyen Duygu")

        result = {
            'prediction': duygu
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)