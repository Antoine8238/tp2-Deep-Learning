import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

# Configuration MLflow
mlflow.set_experiment("TP2_Optimizer_Comparison")

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

# Définition des optimiseurs à comparer
optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

# Dictionnaire pour stocker les résultats
results = {}

# Boucle d'entraînement pour chaque optimiseur
for opt_name, optimizer in optimizers.items():
    print(f"\n{'='*60}")
    print(f"Entraînement avec l'optimiseur: {opt_name}")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"Optimizer_{opt_name}"):
        # Créer un nouveau modèle pour chaque optimiseur
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compiler le modèle avec l'optimiseur courant
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entraîner le modèle
        history = model.fit(
            x_train,
            y_train,
            epochs=15,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Évaluer sur l'ensemble de test
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Sauvegarder les résultats
        results[opt_name] = {
            'history': history,
            'test_acc': test_acc,
            'test_loss': test_loss
        }
        
        # Log des paramètres et métriques dans MLflow
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_param("epochs", 15)
        mlflow.log_param("batch_size", 128)
        
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Log de l'évolution des métriques epoch par epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Log du modèle
        mlflow.keras.log_model(model, f"model_{opt_name}")
        
        print(f"\n📊 Résultats pour {opt_name}:")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")

# Visualisation comparative
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

colors = ['blue', 'green', 'red', 'orange']

# Loss d'entraînement
for (opt_name, result), color in zip(results.items(), colors):
    axes[0, 0].plot(result['history'].history['loss'], 
                    label=opt_name, color=color)
axes[0, 0].set_title('Train Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss de validation
for (opt_name, result), color in zip(results.items(), colors):
    axes[0, 1].plot(result['history'].history['val_loss'], 
                    label=opt_name, color=color)
axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy d'entraînement
for (opt_name, result), color in zip(results.items(), colors):
    axes[1, 0].plot(result['history'].history['accuracy'], 
                    label=opt_name, color=color)
axes[1, 0].set_title('Train Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy de validation
for (opt_name, result), color in zip(results.items(), colors):
    axes[1, 1].plot(result['history'].history['val_accuracy'], 
                    label=opt_name, color=color)
axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Tableau récapitulatif
print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*80)
print(f"{'Optimiseur':<20} {'Test Acc':<12} {'Test Loss':<12} {'Val Acc':<12}")
print("-"*80)
for opt_name, result in results.items():
    val_acc = result['history'].history['val_accuracy'][-1]
    print(f"{opt_name:<20} {result['test_acc']:<12.4f} {result['test_loss']:<12.4f} {val_acc:<12.4f}")

# Meilleur optimiseur
best_optimizer = max(results.items(), key=lambda x: x[1]['test_acc'])
print("\n🏆 MEILLEUR OPTIMISEUR:")
print(f"   {best_optimizer[0]} avec une accuracy de test de {best_optimizer[1]['test_acc']:.4f}")

print("\n💡 OBSERVATIONS:")
print("   • Adam combine les avantages de Momentum et RMSprop")
print("   • SGD simple est généralement plus lent à converger")
print("   • SGD avec momentum améliore significativement la convergence")
print("   • RMSprop adapte le learning rate pour chaque paramètre")