import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os



# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")


# Encoder les variables catégoriques en nombres
# Le modèle ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")    # (500,)


# Découpage des données : 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,    # Proportion du jeu de test (20%)
    random_state=42,  # Garantit la reproductibilité des résultats
    stratify=y       # Maintient la proportion des classes (distribution de y)
)

# Affichage des dimensions pour vérification
print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test         : {X_test.shape[0]} patients")


# Initialisation du modèle Forêt Aléatoire
model = RandomForestClassifier(
    n_estimators=100,  # Nombre d'arbres de décision dans la forêt
    random_state=42    # Garantit des résultats identiques à chaque exécution
)

# Entraînement du modèle sur les données d'apprentissage
model.fit(X_train, y_train)

# Affichage des informations clés du modèle
print("Modèle entraîné avec succès !")
print(f"Nombre d'arbres   : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes détectées  : {list(model.classes_)}")



# 1. Prédire sur les données de test
y_pred = model.predict(X_test)

# 2. Créer un DataFrame pour comparer les 10 premières prédictions avec la réalité
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})

# 3. Affichage du résultat
print("Comparaison (Échantillon de 10 patients) :")
print(comparison)


accuracy = accuracy_score(y_test, y_pred)
print( f"Accuracy : {accuracy :.2%}")


# Matrice de confusion
cm = confusion_matrix( y_test , y_pred , labels = model.classes_ )
print ("Matrice de confusion : " )
print (cm)
# Rapport de classification
print ( "\nRapport de classification : " )
print (classification_report ( y_test , y_pred ) )

# Visualiser avec seaborn
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')

plt.tight_layout()

# Sauvegarde et affichage
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("Figure sauvegardee dans figures/confusion_matrix.png")


# Creer le dossier models / s ' il n ' existe pas
os.makedirs("models" , exist_ok = True )
# Serialiser le modele
joblib.dump(model,"models/model.pkl")
# Verifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs ( indispensables pour les nouvelles donnees )
joblib.dump ( le_sexe , "models/encoder_sexe.pkl")
joblib.dump ( le_region , "models/encoder_region.pkl")
# Sauvegarder la liste des features ( pour reference )
joblib.dump ( feature_cols , "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")

import joblib

# Simuler ce que fera l'API en Lab 3 :
# Charger le modèle DEPUIS LE FICHIER (pas depuis la mémoire)

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Un nouveau patient arrive au centre de santé de Médina
nouveau_patient = {'age': 19,
    'sexe': 'M',
    'temperature': 36.7,
    'tension_sys': 115,
    'toux': False,
    'fatigue': False,
    'maux_tete': False,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:8s} : {proba:.1%} {bar}")

importances = model.feature_importances_

for name, imp in sorted(
    zip(feature_cols, importances),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:20s} : {imp:.3f}")

patients = [
    {
        'age': 19,
        'sexe': 'M',
        'temperature': 36.7,
        'tension_sys': 115,
        'toux': False,
        'fatigue': False,
        'maux_tete': False,
        'region': 'Dakar'
    },
    {
        'age': 35,
        'sexe': 'F',
        'temperature': 39.5,
        'tension_sys': 110,
        'toux': True,
        'fatigue': True,
        'maux_tete': True,
        'region': 'Dakar'
    },
    {
        'age': 72,
        'sexe': 'M',
        'temperature': 37.4,
        'tension_sys': 130,
        'toux': True,
        'fatigue': False,
        'maux_tete': False,
        'region': 'Dakar'
    }
]

for i, patient in enumerate(patients, 1):
    sexe_enc = le_sexe_loaded.transform([patient['sexe']])[0]
    region_enc = le_region_loaded.transform([patient['region']])[0]

    features = [
        patient['age'],
        sexe_enc,
        patient['temperature'],
        patient['tension_sys'],
        int(patient['toux']),
        int(patient['fatigue']),
        int(patient['maux_tete']),
        region_enc
    ]

    diagnostic = model_loaded.predict([features])[0]
    probas = model_loaded.predict_proba([features])[0]

    print(f"\n===== Patient {i} =====")
    print(f"Age : {patient['age']} | Sexe : {patient['sexe']}")
    print(f"Température : {patient['temperature']}°C")
    print(f"Diagnostic : {diagnostic}")

    print("Probabilités :")
    for classe, proba in zip(model_loaded.classes_, probas):
        print(f"{classe:10s} : {proba:.1%}")