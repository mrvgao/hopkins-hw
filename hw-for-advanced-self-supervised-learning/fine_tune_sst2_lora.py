import torch
import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import wandb

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the SST2 dataset from 'stanfordnlp/sst2'
ds = load_dataset("stanfordnlp/sst2")

# Load the tokenizer and model, and move the model to the device
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# LoRA configuration
lora_config = LoraConfig(
    r=8,                        # Low-rank parameter (this defines the rank)
    lora_alpha=32,               # Scaling factor
    target_modules=["query", "value"],  # Apply LoRA to attention layers (query, value in this case)
    lora_dropout=0.1,            # Dropout in the LoRA layers
    bias="none",                 # Whether to train bias terms
    task_type="SEQ_CLS"          # Task type is sequence classification
)

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.to(device)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = ds.map(tokenize_function, batched=True)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute accuracy
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Training arguments
output_dir = "./results-lora-sst2"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",        # Save model at the end of every epoch
    learning_rate=1e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,  # Load the best model at the end of training
    save_total_limit=1            # Keep only the best model saved
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

if __name__ == '__main__':
    # Train the model
    train_result = trainer.train()

    # Save the model
    trainer.save_model("./roberta-lora-sst2")

    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)

    # Log metrics
    train_metrics = trainer.state.log_history

    # Extract accuracy values for plotting
    train_accuracy = []
    eval_accuracy = []
    epoch_train = []
    epoch_val = []
    for log in train_metrics:
        if 'eval_accuracy' in log:
            eval_accuracy.append(log['eval_accuracy'])
            epoch_val.append(log['epoch'])
        if 'train_accuracy' in log:
            train_accuracy.append(log['train_accuracy'])
            epoch_train.append(log['epoch'])

    # Plot train and eval accuracy

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_train, train_accuracy, label='Train Accuracy', marker='o')
    plt.plot(epoch_val, eval_accuracy, label='Validation Accuracy', marker='o')
    plt.title('RoBERTa Base SST2 with LoRA - Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot_LoRA.png')

    # Optionally, clear the figure after saving
    plt.clf()  # Clears the current figure to free memory