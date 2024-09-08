from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import random_split

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def tokenize_function(example):
    return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128)
# Load the SST2 dataset from 'stanfordnlp/sst2'
ds = load_dataset("stanfordnlp/sst2")

# Load the tokenizer and model, and move the model to the device
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)

# Tokenize the dataset

tokenized_datasets = ds.map(tokenize_function, batched=True)

train_size = len(tokenized_datasets["train"])
eval_train_size = int(0.2 * train_size)
train_size = train_size - eval_train_size
train_dataset, val_dataset = random_split(tokenized_datasets["train"], [train_size, eval_train_size])
test_dataset = tokenized_datasets["validation"]

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Training arguments
# Compute accuracy
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_eval_loop(trainer, num_train_epochs, _train_dataset, _val_dataset, _test_dataset, title, save_path):
    best_model_state_dict = None  # This will hold the best model's state_dict in memory
    best_val_accuracy = 0.0

    train_accuracies = []
    val_accuracies = []
    epoch_train = []

    early_stop_patience = 3  # Stop if validation accuracy doesn't improve after 3 epochs
    early_stop_counter = 0  # Counter to track consecutive epochs with no improvement

    for epoch in range(int(num_train_epochs)):
        print(f"Epoch {epoch + 1}/{num_train_epochs}")

        trainer.train()

        train_results = trainer.evaluate(eval_dataset=_train_dataset)
        train_accuracy = train_results['eval_accuracy']
        train_accuracies.append(train_accuracy)
        epoch_train.append(epoch + 1)
        print(f"Training evaluation accuracy : {train_accuracy:.4f}")

        # Evaluate on the separate validation set
        val_results = trainer.evaluate(eval_dataset=_val_dataset)
        val_accuracy = val_results['eval_accuracy']
        val_accuracies.append(val_accuracy)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving the model.")
            best_model_state_dict = trainer.model.state_dict()  # Save the model state_dict in memory
            early_stop_counter = 0
        else:
            early_stop_counter += 1  # Increment early stopping counter if no improvement

        if early_stop_counter >= early_stop_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs with best validation accuracy: {best_val_accuracy:.4f}")
            break  # Exit the training loop early if no improvement

    print('training accuracy:', train_accuracies)
    print('validation accuracy:', val_accuracies)

    # Load the best model and evaluate on the test dataset
    trainer.model.load_state_dict(best_model_state_dict)

    # Evaluate on the test dataset
    test_results = trainer.evaluate(eval_dataset=_test_dataset)
    test_accuracy = test_results['eval_accuracy']
    print(f"Final test accuracy with the best model: {test_accuracy:.4f}")

    # Plot the train vs eval accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_train, train_accuracies, label='Train Accuracy', marker='+')
    plt.plot(epoch_train, val_accuracies, label='Validation Accuracy', marker='o')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

    # Optionally, clear the figure after saving
    plt.clf()  # Clears the current figure to free memory
