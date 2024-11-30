from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM

# Load the dataset
dataset = load_dataset('json', data_files={
    'train': r'C:\\Users\\HP pavilion\\OneDrive\\Desktop\\Agroxpert assist\\train1.json', 
    'test': r'C:\\Users\\HP pavilion\\OneDrive\\Desktop\\Agroxpert assist\\test.json'
})

# Verify dataset structure
print(dataset)

# Load GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token to EOS token for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    combined_text = [f"Q: {q} A: {a}" for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(combined_text, padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set PyTorch tensor format
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])
print(tokenized_datasets['train'][0])  # Debugging tokenized data


# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Evaluate after each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=7,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    save_strategy="epoch",  # Save checkpoint after each epoch
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)

# Train the Model
trainer.train()

# Save the Model and Tokenizer
model.save_pretrained(r"C:\\Users\\HP pavilion\\OneDrive\\Desktop\\Agroxpert assist\\model-folder")
tokenizer.save_pretrained(r"C:\\Users\\HP pavilion\\OneDrive\\Desktop\\Agroxpert assist\\model-folder")

# Load the Fine-Tuned Model and Tokenizer
model_directory = r"C:\\Users\\HP pavilion\\OneDrive\\Desktop\\Agroxpert assist\\model-folder"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

print("Model is ready for inference or further fine-tuning!")
