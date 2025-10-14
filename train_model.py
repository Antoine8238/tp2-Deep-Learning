import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

print("🚀 Début de l'entraînement du modèle MNIST...")

# Variables pour les paramètres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Chargement du jeu de données MNIST
print("📥 Chargement des données MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Division en ensembles train/validation (90% train, 10% validation)
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation des données
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# One-hot encoding des labels
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"Taille ensemble d'entraînement: {x_train.shape}")
print(f"Taille ensemble de validation: {x_val.shape}")
print(f"Taille ensemble de test: {x_test.shape}")

# Lancement du tracking MLflow
print("\n📊 Démarrage du tracking MLflow...")
with mlflow.start_run():
    # Enregistrement des paramètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    
    # Construction du modèle
    print("\n🏗️  Construction du modèle...")
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compilation (corrigée)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # ✅ Corrigé ici
        metrics=['accuracy']
    )
    
    print("✅ Modèle construit et compilé")
    model.summary()
    
    # Entraînement
    print(f"\n🎯 Entraînement du modèle ({EPOCHS} epochs)...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Évaluation sur l'ensemble de test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nPerformance sur l'ensemble de test:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    
    # Courbe d'Accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Courbes d\'Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bias_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Diagnostic Bias/Variance
    print("\n=== DIAGNOSTIC BIAS/VARIANCE ===")
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"Train Loss finale: {final_train_loss:.4f}")
    print(f"Validation Loss finale: {final_val_loss:.4f}")
    print(f"Train Accuracy finale: {final_train_acc:.4f}")
    print(f"Validation Accuracy finale: {final_val_acc:.4f}")

    # Analyse automatique
    if final_train_acc < 0.90:
        print("\n⚠️  HIGH BIAS (Underfitting) détecté:")
        print("   - Le modèle performe mal même sur les données d'entraînement")
        print("   - Solutions: augmenter la capacité du modèle, entraîner plus longtemps")
    elif final_val_acc < final_train_acc - 0.05:
        print("\n⚠️  HIGH VARIANCE (Overfitting) détecté:")
        print("   - Grande différence entre performance train et validation")
        print("   - Solutions: régularisation, dropout, plus de données")
    else:
        print("\n✅ Le modèle semble bien équilibré")
