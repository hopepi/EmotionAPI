import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
import pandas as pd
import re

# Veriyi yükle
df = pd.read_csv("text.csv")

X = df['text']
Y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Özel karakterleri kaldırma
    return text

# Eğitim verisini temizle
X_train_clean = [clean_text(text) for text in X_train]

# Tokenizer'ı oluştur ve eğit
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_clean)

# Tokenizer'ı kaydet
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Eğitim ve test setlerini tokenlara dönüştür
X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Dizileri sabit uzunlukta doldur
max_len = 100  # Dizilerin maksimum uzunluğu
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Etiketleri sayısal değerlere çevir
le = LabelEncoder()
y_train_num = le.fit_transform(y_train)
y_test_num = le.transform(y_test)

# One-hot encoding
y_train_cat = to_categorical(y_train_num)
y_test_cat = to_categorical(y_test_num)

# Modeli oluştur
model = Sequential()

# Gömme (Embedding) katmanı
model.add(Embedding(input_dim=15000, output_dim=256, input_length=max_len))

# LSTM Katmanı (GRU Katmanı için GRU() kullan)
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))

# Dropout ile overfitting'i azaltma
model.add(Dropout(0.3))

# Tam bağlantılı (Dense) katman, duygu sayısı kadar çıktı
model.add(Dense(6, activation='softmax'))  # 6 duygu için


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train_pad, y_train_cat, epochs=10, batch_size=64,
                    validation_data=(X_test_pad, y_test_cat),
                    callbacks=[early_stopping])

# Modeli ve tokenizer'ı kaydet
model.save("model.h5")
