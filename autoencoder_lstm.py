import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler

# Exemple de données : séquences de prix fictifs
prices = [0.05, 0.07, 0.10, 0.15, 0.20, 0.18, 0.17, 0.19, 0.21, 0.22]
prices = np.array(prices).reshape(-1, 1)

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Préparation des séquences pour LSTM
X = []
for i in range(len(prices_scaled) - 3):
    X.append(prices_scaled[i:i+3])
X = np.array(X)

# Définition du modèle Autoencodeur LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    RepeatVector(X.shape[1]),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer='adam', loss='mse')

# Entraînement du modèle
model.fit(X, X, epochs=50, batch_size=1, verbose=1)

# Prédiction et évaluation des erreurs de reconstruction
X_pred = model.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
print("Erreur moyenne de reconstruction : ", np.mean(
