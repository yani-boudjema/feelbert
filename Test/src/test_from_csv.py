import pandas as pd
from transformers import pipeline

# === 1. Charger les 10 phrases depuis le fichier CSV ===
df = pd.read_csv("./../Data/test_english_sentences.csv")
phrases = df["text"].tolist()

# === 2. Charger le modèle fine-tuné ===
classifier = pipeline(
    "text-classification",
    model="./finetuned_bert_advanced",
    tokenizer="./finetuned_bert_advanced"
)

# === 3. Tester chaque phrase ===
for phrase in phrases:
    prediction = classifier(phrase)
    print(f"Phrase : {phrase}\nRésultat : {prediction}\n")
