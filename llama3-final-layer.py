import os
import torch
import csv
from transformers import AutoTokenizer, AutoModel

# Load Llama3 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Function to get the embedding for a target word using Llama3 model
def get_word_embedding(target_word):
    inputs = tokenizer(target_word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Function to process a single CSV file and get embeddings for the target words
def process_file(file_path, language):
    embeddings = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip header if present
        for row in csvreader:
            target_word = row[2]  # Assuming the target word is in the fourth column
            embedding = get_word_embedding(target_word)
            embeddings.append({
                'target_word': target_word,
                'embedding': embedding
            })
    return {language: embeddings}

# Function to save word embeddings to a CSV file
def save_embeddings_csv(language_embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for language, data in language_embeddings.items():
        output_file = os.path.join(output_dir, f'{language}_target_word_embeddings.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['target_word'] + [f'embedding_{i}' for i in range(len(data[0]['embedding']))])
            for entry in data:
                csvwriter.writerow([entry['target_word']] + entry['embedding'])

# Function to process all CSV files in a directory, assuming each file represents a different language
def process_language_files(directory):
    all_language_embeddings = {}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            language = file.split('.')[0]  # Assuming file name format is 'language_code.csv'
            language_embeddings = process_file(file_path, language)
            all_language_embeddings.update(language_embeddings)
    return all_language_embeddings

# Specify the directory containing your CSV files
input_directory = 'Languages'

# # Process all CSV files in the directory
all_language_embeddings = process_language_files(input_directory)

# # Specify the output directory to save word embeddings
output_directory = 'WordEmbeddingsUpdated'

# # Save word embeddings to the output directory in CSV format
save_embeddings_csv(all_language_embeddings, output_directory)


