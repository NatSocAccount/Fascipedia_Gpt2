#pip install transformers

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print("Model and Tokenizer loaded successfully!")

# Load your dataset
data_files = {'train': 'path_to/italy_data.txt'}
dataset = load_dataset('text', data_files=data_files)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-italy",
    overwrite_output_dir=True,
    num_train_epochs=3,    # adjust the number of epochs
    per_device_train_batch_size=2,  # batch size depends on GPU/CPU memory
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-italy-model")
