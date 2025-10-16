import numpy as np
import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt

# Chargement et préparation des données (même code que Ex1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Fonction pour créer et entraîner un modèle
def train_model(use_regularization=False, model_name="Base"):
    print(f"\n{'='*50}")
    print(f"Entraînement du modèle: {model_name}")
    print(f"{'='*50}")
    
    if use_regularization:
        # Modèle AVEC régularisation
        model = keras.Sequential([
            layers.Dense(
                512, 
                activation='relu', 
                input_shape=(784,),
                kernel_regularizer=regularizers.l2(0.001)  # L2 régularisation
            ),
            layers.Dropout(0.5),  # Dropout après la couche d'entrée
            layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.3),  # Dropout avant la sortie
            layers.Dense(10, activation='softmax')
        ])
    else:
        # Modèle SANS régularisation
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    return history, test_acc

# Entraîner les deux modèles
history_base, test_acc_base = train_model(
    use_regularization=False, 
    model_name="Sans Régularisation"
)
history_reg, test_acc_reg = train_model(
    use_regularization=True, 
    model_name="Avec Régularisation (L2 + Dropout)"
)

# Visualisation comparative
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss - Modèle sans régularisation
axes[0, 0].plot(history_base.history['loss'], label='Train Loss')
axes[0, 0].plot(history_base.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Loss - Sans Régularisation')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss - Modèle avec régularisation
axes[0, 1].plot(history_reg.history['loss'], label='Train Loss', color='green')
axes[0, 1].plot(history_reg.history['val_loss'], label='Val Loss', color='red')
axes[0, 1].set_title('Loss - Avec Régularisation')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Accuracy - Modèle sans régularisation
axes[1, 0].plot(history_base.history['accuracy'], label='Train Acc')
axes[1, 0].plot(history_base.history['val_accuracy'], label='Val Acc')
axes[1, 0].set_title('Accuracy - Sans Régularisation')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Accuracy - Modèle avec régularisation
axes[1, 1].plot(history_reg.history['accuracy'], label='Train Acc', color='green')
axes[1, 1].plot(history_reg.history['val_accuracy'], label='Val Acc', color='red')
axes[1, 1].set_title('Accuracy - Avec Régularisation')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Résumé des résultats
print("\n" + "="*60)
print("RÉSUMÉ DES RÉSULTATS")
print("="*60)

print("\nModèle SANS régularisation:")
print(f"  Train Accuracy: {history_base.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy: {history_base.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy: {test_acc_base:.4f}")
print(f"  Écart Train-Val: {history_base.history['accuracy'][-1] - history_base.history['val_accuracy'][-1]:.4f}")

print("\nModèle AVEC régularisation:")
print(f"  Train Accuracy: {history_reg.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy: {history_reg.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy: {test_acc_reg:.4f}")
print(f"  Écart Train-Val: {history_reg.history['accuracy'][-1] - history_reg.history['val_accuracy'][-1]:.4f}")

print("\n💡 ANALYSE:")
if history_reg.history['val_accuracy'][-1] > history_base.history['val_accuracy'][-1]:
    print("✅ La régularisation a AMÉLIORÉ les performances sur la validation")
    print("   Réduction de l'overfitting observée")
else:
    print("⚠️  La régularisation n'a pas amélioré les performances")
    print("   Le modèle de base ne souffrait peut-être pas d'overfitting")