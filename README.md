#  TP2 – Amélioration des Réseaux de Neurones Profonds

**Auteur** : Antoine Emmanuel ESSOMBA ESSOMBA
**Encadrant** : Dr Louis Fippo
**Département** : Génie Informatique, ENSPY  


##  Objectifs pédagogiques

- Diagnostiquer les problèmes de biais et de variance.
- Appliquer les techniques de régularisation : L2 et Dropout.
- Accélérer l'entraînement avec la normalisation par lot (Batch Normalization).
- Comparer les performances des algorithmes d'optimisation : SGD, RMSprop, Adam.
- Utiliser MLflow pour le suivi expérimental.

---


---

##  Partie  – Théorie

- **Biais vs. Variance** : Analyse des erreurs d'entraînement et de validation.
- **Régularisation** : L2 (weight decay), Dropout.
- **Normalisation** : Batch Normalization pour stabiliser l'entraînement.
- **Optimisation** : Comparaison entre SGD (avec momentum), RMSprop et Adam.

---

##  Partie  – Implémentation

###  1. Analyse Biais/Variance
- Chargement de MNIST avec séparation explicite : 90% entraînement, 10% validation.
- Observation des courbes de perte et d'exactitude.
- Diagnostic du comportement du modèle.

###  2. Régularisation
- Ajout de L2 sur les couches denses : `kernel_regularizer=keras.regularizers.l2(0.001)`
- Ajout de Dropout après la couche d'entrée.
- Comparaison des performances avec/sans régularisation.

###  3. Optimisation
- Boucle d'entraînement avec 3 optimisateurs :
  - `SGD` avec momentum
  - `RMSprop`
  - `Adam`
- Suivi des expériences avec **MLflow** :
  - `mlflow.log_param("optimizer", ...)`
  - `mlflow.log_metric("final_test_accuracy", ...)`

###  4. Batch Normalization
- Ajout de `BatchNormalization()` entre la couche dense et Dropout.
- Comparaison de la vitesse de convergence.

---

##  Suivi avec MLflow

Lancement de l’interface :

```bash
mlflow ui
