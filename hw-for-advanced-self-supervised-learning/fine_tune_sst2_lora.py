from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
import os
from hw_utilitis import compute_metrics, data_collator, tokenizer, train_dataset, val_dataset, test_dataset, model
# Check if GPU is available and set the device

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

# Training arguments
output_dir = "./results-lora-sst2"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=1            # Keep only the best model saved
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    # Train the model
    from hw_utilitis import train_eval_loop

    train_eval_loop(trainer, 5, train_dataset, val_dataset, test_dataset, 'RoBERTa LoRA SST2', 'accuracy_plot_LoRA.png')