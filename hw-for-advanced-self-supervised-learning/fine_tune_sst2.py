from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import os
from hw_utilitis import compute_metrics, data_collator, tokenizer, train_dataset, val_dataset, test_dataset, model


output_dir = "./results-whole"
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
    # train_dataset=tokenized_datasets["train"],
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if __name__ == '__main__':
    # Train the model
    # Training loopeval_train_subset
    from hw_utilitis import train_eval_loop

    train_eval_loop(trainer, 5, train_dataset, val_dataset, test_dataset, 'RoBERTa Base SST2', 'accuracy_plot_Whole.png')
