"""import tensorflow as tf
import numpy as np
from flask import Flask,request,jsonify
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import re
from keras.src.utils import pad_sequences
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.saving import load_model
import pickle
import gunicorn
print(gunicorn.__version__)
print(tf.__version__)

model = load_model("model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

#TEXT TEMİZLEME
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

        text_seq = tokenizer.texts_to_sequences([cleaned_input])
        text_pad = pad_sequences(text_seq, maxlen=100, padding='post')

        prediction = model.predict(text_pad)
        predicted_label = prediction.argmax()
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
    app.run(debug=True)