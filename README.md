# Fine-Tuning personnalisé de BERT

## 📌 Description du projet

Ce projet a pour objectif de réaliser un **fine-tuning de modèles de
type BERT** pour l'analyse de sentiments, en utilisant un dataset
personnalisé.\
Plusieurs versions et expérimentations ont été menées afin d'améliorer
la performance et de comparer différents modèles.

## 🚀 Étapes principales

### 🔹 Fine-Tuning V1

-   BERT est fine-tuné sur un dataset personnalisé.\
-   Première version basique pour établir une ligne de référence.

### 🔹 Fine-Tuning V2

-   Ajout d'une **classe "neutre"** dans le dataset.\
-   Mise en place de la stratégie **Early Stopping** pour éviter
    l'overfitting.

### 🔹 Fine-Tuning V3

-   Enrichissement du dataset (plus de données, meilleure couverture).\
-   Maintien de l'**Early Stopping** pour optimiser l'entraînement.

### 🔹 Comparaison avec RoBERTa

-   RoBERTa est fine-tuné et comparé à BERT.\
-   Objectif : mesurer l'impact de l'architecture sur la performance.

### 🔹 Test de CamemBERT

-   CamemBERT est évalué sur des données traduites.\
-   Permet de tester la robustesse et les performances d'un modèle
    adapté au français.

## 🛠️ Technologies utilisées

-   [Transformers (Hugging Face)](https://huggingface.co/transformers/)\
-   PyTorch\
-   Pandas\
-   Scikit-learn\
-   Logging Python

## 📊 Résultats attendus

-   Amélioration progressive du modèle via différentes étapes de
    fine-tuning\
-   Comparaison entre BERT, RoBERTa et CamemBERT\
-   Analyse des performances en fonction des versions (V1, V2, V3)

## 🔮 Prochaines étapes

-   Optimiser davantage l'entraînement avec une meilleure recherche
    d'hyperparamètres\
-   Tester d'autres modèles multilingues ou spécialisés\
-   Mettre en place une API pour exposer le modèle fine-tuné
