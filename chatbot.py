# Import required libraries
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Step 1: Load the local dataset
print("Loading dataset...")
data = pd.read_csv("data.txt", header=None, names=["text"])  # Load plain text file
dataset = Dataset.from_pandas(data)

# Verify the dataset
print("Dataset Sample:")
print(data.head())

# Step 2: Load the GPT-2 tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Use input_ids as labels
    return tokenized

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 3: Load the GPT-2 model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Step 4: Set up the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # Directory to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=4,  # Batch size
    save_steps=500,  # Save the model every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,  # Log every 10 steps
    prediction_loss_only=True,  # Log only the loss
)

# Step 5: Define the trainer
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the fine-tuned model
print("Saving model...")
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Fine-tuning complete! The model is saved in './gpt2-finetuned'.")