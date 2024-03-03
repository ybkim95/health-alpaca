import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load your custom dataset
path_to_json_file = 'path/to/your/file.json'

def load_custom_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

# Prepare the dataset
data = load_custom_data(path_to_json_file)
dataset = Dataset.from_dict({
    'input_text': [f"Answer this question truthfully: {x['input']}" for x in data],  # Prepend instruction to input
    'output_text': [x['output'] for x in data],
})

# Load tokenizer and model
model_checkpoint = "medalpaca/medalpaca-13b"  # Ensure this is the correct model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Tokenize the dataset
def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['output_text'], max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,   # Adjust based on your GPU memory
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
