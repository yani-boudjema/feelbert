from transformers import pipeline

# Charger le modèle de sentiment standard (POSITIVE / NEGATIVE)
classifier = pipeline("sentiment-analysis")

# Liste de phrases avec ironie, sarcasme, ambiguïté
phrases = [
    "Oh great, another rainy Monday... exactly what I needed.",
    "He said 'great job' after I completely messed up the presentation.",
    "Well, at least the food was cold and tasteless. Just what I wanted!",
    "Sure, being stuck in traffic for 3 hours is my idea of fun.",
    "I guess getting ignored all day means I'm really popular.",
    "The product broke after one use, but hey, it looked nice in the trash.",
    "The movie was... something else. Not sure what I watched.",
    "Thanks for the support, really felt like I was all alone out there.",
    "Amazing service! They lost my luggage and hung up on me.",
    "What a wonderful day to lose my keys and miss the train."
]

# Appliquer le modèle à chaque phrase
for phrase in phrases:
    result = classifier(phrase)
    print(f"Phrase: {phrase}")
    print(f"Résultat: {result}\n")
