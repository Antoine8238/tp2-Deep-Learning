import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import time

# Chargement et préparation des données
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
def train_model(use_batch_norm=False, model_name="Base"):
    print(f"\n{'='*60}")
    print(f"Entraînement: {model_name}")
    print(f"{'='*60}")
    
    if use_batch_norm:
        # Modèle AVEC Batch Normalization
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.BatchNormalization(),  # BN après Dense
            layers.Dropout(0.2),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),  # BN après Dense
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),  # BN après Dense
            layers.Dropout(0.2),
            
            layers.Dense(10, activation='softmax')
        ])
    else:
        # Modèle SANS Batch Normalization
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Afficher l'architecture
    print(f"\nArchitecture du modèle:")
    model.summary()
    
    # Mesurer le temps d'entraînement
    start_time = time.time()
    
    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    return history, test_acc, training_time, model

# Entraîner les deux modèles
print("🔵 PHASE 1: Modèle sans Batch Normalization")
history_base, test_acc_base, time_base, model_base = train_model(
    use_batch_norm=False, 
    model_name="Sans Batch Normalization"
)

print("\n🟢 PHASE 2: Modèle avec Batch Normalization")
history_bn, test_acc_bn, time_bn, model_bn = train_model(
    use_batch_norm=True, 
    model_name="Avec Batch Normalization"
)

# Visualisation comparative détaillée
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss - Sans BN
axes[0, 0].plot(history_base.history['loss'], label='Train', color='blue')
axes[0, 0].plot(history_base.history['val_loss'], label='Validation', color='red')
axes[0, 0].set_title('Loss - Sans Batch Normalization', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss - Avec BN
axes[0, 1].plot(history_bn.history['loss'], label='Train', color='green')
axes[0, 1].plot(history_bn.history['val_loss'], label='Validation', color='orange')
axes[0, 1].set_title('Loss - Avec Batch Normalization', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Comparaison directe Loss
axes[0, 2].plot(history_base.history['val_loss'], label='Sans BN', color='red', linewidth=2)
axes[0, 2].plot(history_bn.history['val_loss'], label='Avec BN', color='orange', linewidth=2)
axes[0, 2].set_title('Comparaison Validation Loss', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Accuracy - Sans BN
axes[1, 0].plot(history_base.history['accuracy'], label='Train', color='blue')
axes[1, 0].plot(history_base.history['val_accuracy'], label='Validation', color='red')
axes[1, 0].set_title('Accuracy - Sans Batch Normalization', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy - Avec BN
axes[1, 1].plot(history_bn.history['accuracy'], label='Train', color='green')
axes[1, 1].plot(history_bn.history['val_accuracy'], label='Validation', color='orange')
axes[1, 1].set_title('Accuracy - Avec Batch Normalization', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Comparaison directe Accuracy
axes[1, 2].plot(history_base.history['val_accuracy'], label='Sans BN', color='red', linewidth=2)
axes[1, 2].plot(history_bn.history['val_accuracy'], label='Avec BN', color='orange', linewidth=2)
axes[1, 2].set_title('Comparaison Validation Accuracy', fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('batch_normalization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyse de la vitesse de convergence
print("\n" + "="*80)
print("ANALYSE DE LA VITESSE DE CONVERGENCE")
print("="*80)

# Trouver l'epoch où le modèle atteint 95% d'accuracy sur la validation
def find_convergence_epoch(history, threshold=0.95):
    val_acc = history.history['val_accuracy']
    for epoch, acc in enumerate(val_acc):
        if acc >= threshold:
            return epoch + 1
    return None

conv_epoch_base = find_convergence_epoch(history_base)
conv_epoch_bn = find_convergence_epoch(history_bn)

print("\nTemps pour atteindre 95% accuracy sur validation:")
if conv_epoch_base:
    print(f"  Sans BN: Epoch {conv_epoch_base}")
else:
    print(f"  Sans BN: N'a pas atteint 95%")
    
if conv_epoch_bn:
    print(f"  Avec BN: Epoch {conv_epoch_bn}")
else:
    print(f"  Avec BN: N'a pas atteint 95%")

if conv_epoch_base and conv_epoch_bn:
    improvement = ((conv_epoch_base - conv_epoch_bn) / conv_epoch_base) * 100
    print(f"\n  ⚡ Accélération: {improvement:.1f}% plus rapide avec Batch Normalization")

# Résumé détaillé
print("\n" + "="*80)
print("RÉSUMÉ DÉTAILLÉ DES PERFORMANCES")
print("="*80)

print("\n📊 Sans Batch Normalization:")
print(f"  Train Accuracy finale:      {history_base.history['accuracy'][-1]:.4f}")
print(f"  Validation Accuracy finale: {history_base.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:              {test_acc_base:.4f}")
print(f"  Train Loss finale:          {history_base.history['loss'][-1]:.4f}")
print(f"  Validation Loss finale:     {history_base.history['val_loss'][-1]:.4f}")
print(f"  Temps d'entraînement:       {time_base:.2f}s")
print(f"  Nombre de paramètres:       {model_base.count_params()}")

print("\n📊 Avec Batch Normalization:")
print(f"  Train Accuracy finale:      {history_bn.history['accuracy'][-1]:.4f}")
print(f"  Validation Accuracy finale: {history_bn.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:              {test_acc_bn:.4f}")
print(f"  Train Loss finale:          {history_bn.history['loss'][-1]:.4f}")
print(f"  Validation Loss finale:     {history_bn.history['val_loss'][-1]:.4f}")
print(f"  Temps d'entraînement:       {time_bn:.2f}s")
print(f"  Nombre de paramètres:       {model_bn.count_params()}")

# Amélioration
print("\n🎯 AMÉLIORATIONS APPORTÉES PAR BATCH NORMALIZATION:")
acc_improvement = (test_acc_bn - test_acc_base) * 100
print(f"  Amélioration Test Accuracy: {acc_improvement:+.2f}%")

time_diff = ((time_bn - time_base) / time_base) * 100
print(f"  Différence temps d'entraînement: {time_diff:+.1f}%")

params_increase = model_bn.count_params() - model_base.count_params()
print(f"  Augmentation paramètres: +{params_increase} paramètres")

print("\n💡 AVANTAGES DE BATCH NORMALIZATION:")
print("  ✅ Stabilise l'entraînement en normalisant les activations")
print("  ✅ Permet d'utiliser des learning rates plus élevés")
print("  ✅ Réduit la sensibilité à l'initialisation des poids")
print("  ✅ A un effet régularisant (réduit le besoin de dropout)")
print("  ✅ Accélère la convergence du modèle")

print("\n⚠️  CONSIDÉRATIONS:")
print("  • Ajoute légèrement au temps de calcul par epoch")
print("  • Augmente le nombre de paramètres à entraîner")
print("  • Comportement différent entre entraînement et inférence")