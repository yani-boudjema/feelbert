from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import logging

# Configuration du logging pour voir les erreurs et les avertissements
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(csv_file):
    try:
        # Charger les données
        logger.info(f"Chargement des données depuis {csv_file}")
        df = pd.read_csv(csv_file)
        phrases = df["text"].tolist()
        
        # MÉTHODE 3: Utiliser un modèle alternatif multilingue (repli)
        logger.info("Tentative avec modèle BERT multilingue")
        classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Traiter chaque phrase
        for phrase in phrases:
            result = classifier(phrase)
            
            # Convertir le résultat en positif/neutre/négatif
            # Le modèle renvoie généralement des étiquettes comme "1 star", "2 stars", etc.
            label = result[0]["label"]
            score = result[0]["score"]
            
            # Extraire le nombre d'étoiles (normalement de 1 à 5)
            try:
                stars = int(label.split()[0])
                
                # Convertir en sentiment
                if stars >= 4:
                    sentiment = "positif"
                elif stars <= 2:
                    sentiment = "négatif"
                else:
                    sentiment = "neutre"
            except:
                # Au cas où le format serait différent
                if "positive" in label.lower():
                    sentiment = "positif"
                elif "negative" in label.lower():
                    sentiment = "négatif"
                else:
                    sentiment = "neutre"
            
            print(f"Phrase : {phrase}")
            print(f"Sentiment : {sentiment} (score: {score:.4f}, label original: {label})\n")
            
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        print(f"Une erreur s'est produite: {e}")

if __name__ == "__main__":
    # Nom du fichier CSV
    csv_file = "./../Data/enhanced_sentiment_dataset.csv"
    
    # Analyser le sentiment
    analyze_sentiment(csv_file)