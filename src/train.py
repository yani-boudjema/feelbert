from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# === Charger le dataset ===
df = pd.read_csv("./../Data/sentiment_200_binary.csv")

df = df[df["label"].isin(["positive", "negative"])]

label2id = {'negative': 0, 'positive': 1}
df['label'] = df['label'].map(label2id)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
tokenized_dataset = dataset.map(tokenize, batched=True)

# === Modèle
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)

# === Métriques ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# === Entraînement ===
training_args = TrainingArguments(
    output_dir="./results_binary",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs_binary',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Sauvegarde du modèle fine-tuné
trainer.save_model("./finetuned_bert_binary")
tokenizer.save_pretrained("./finetuned_bert_binary")
