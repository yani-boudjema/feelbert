
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score

# === 1. Charger le dataset large ===
df = pd.read_csv("./../Data/train_sentiment_2100.csv")
label2id = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].map(label2id)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# === 2. Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
tokenized_dataset = dataset.map(tokenize, batched=True)

# === 3. Charger le modèle ===
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2}
)

# === 4. Métriques ===
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# === 5. Entraînement avec EarlyStopping ===
training_args = TrainingArguments(
    output_dir="./results_large",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs_large",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# === 6. Sauvegarde du modèle fine-tuné ===
trainer.save_model("./finetuned_bert_large")
tokenizer.save_pretrained("./finetuned_bert_large")
