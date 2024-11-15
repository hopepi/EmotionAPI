import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import re
import pickle
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences

# TensorFlow Lite modelini yükleyin
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Tokenizer'ı yükleyin
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

# TEXT TEMİZLEME
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text_input = data['input']
        cleaned_input = clean_text(text_input)

        # Tokenize ve pad işlemi
        text_seq = tokenizer.texts_to_sequences([cleaned_input])
        text_pad = pad_sequences(text_seq, maxlen=100, padding='post')

        # TensorFlow Lite modelini kullanarak tahmin yapma
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Girdi verisini ayarlama
        interpreter.set_tensor(input_details[0]['index'], text_pad.astype(np.float32))

        # Modeli çalıştır
        interpreter.invoke()

        # Çıktı verisini al
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Tahmin edilen etiket
        predicted_label = np.argmax(output_data)

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
