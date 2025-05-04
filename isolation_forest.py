import pandas as pd
from sklearn.ensemble import IsolationForest

# Exemple de données : volume et prix fictifs
data = {
    'volume': [1000, 2000, 3000, 50000, 1500, 6000, 3500, 150000, 4000, 1200],
    'prix': [0.05, 0.07, 0.10, 0.20, 0.06, 0.08, 0.15, 0.25, 0.09, 0.06]
}

df = pd.DataFrame(data)

# Initialisation du modèle Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(df[['volume', 'prix']])

# Détection des anomalies
df['anomalie'] = model.predict(df[['volume', 'prix']])
df['anomalie'] = df['anomalie'].apply(lambda x: 'Anomalie' if x == -1 else 'Normal')

# Affichage des résultats
print(df)
