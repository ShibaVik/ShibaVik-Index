import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Exemple de données : Tweets fictifs et sentiments associés
texts = [
    "Memecoin will moon!", "Great investment in Dogecoin!", "Stay away from this coin.",
    "Investing in memecoins is risky", "Pump and dump detected", "Solid project, buy now!"
]
labels = [1, 1, 0, 0, 0, 1]  # 1 pour positif, 0 pour négatif

# Prétraitement des textes
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=10)

# Définition du modèle LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=10),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, np.array(labels), epochs=5, batch_size=32)

# Prédiction de sentiments
predictions = model.predict(X)
print("Prédictions de sentiment : ", predictions)
