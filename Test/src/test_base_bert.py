
from transformers import pipeline
import pandas as pd

# Charger le pipeline avec le modèle BERT de base fine-tuné sur SST-2
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Charger les phrases du fichier CSV
df = pd.read_csv("./../Data/test_english_sentences.csv")
phrases = df["text"].tolist()

# Tester chaque phrase
for phrase in phrases:
    result = classifier(phrase)
    print(f"Phrase: {phrase}")
    print(f"Résultat: {result}\n")
