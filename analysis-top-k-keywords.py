import csv

def read_csv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for lang_iso_code, concept_id, ortho_form, raw_ipa, next_step in reader:
            data[concept_id] = ortho_form
    return data

def read_top_similar_words(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(':')
            en_word = parts[0].strip()
            es_words = [word.strip() for word in parts[1].split(',')]
            data[en_word] = es_words
    return data

def calculate_accuracy(es_data, en_data, en_es_data):
    total_concepts = 0
    correct_predictions = 0

    for concept_id, ortho_form_es in es_data.items():
        total_concepts += 1
        en_word = en_data.get(concept_id)  # Get the English word corresponding to the concept_id
        if en_word:
            en_word_translations = en_es_data.get(en_word, [])
            if ortho_form_es in en_word_translations:
                correct_predictions += 1

    accuracy = (correct_predictions / total_concepts) * 100 if total_concepts > 0 else 0
    return accuracy

def process_language(lang):
    es_data = read_csv(f'Languages/{lang}.csv')
    en_data = read_csv('Languages/en.csv')
    en_es_data = read_top_similar_words(f'top_k/top_similar_words_en_{lang}.txt')
    
    accuracy = calculate_accuracy(es_data, en_data, en_es_data)
    return accuracy

def main():
    languages = ['ar', 'es', 'fr', 'he', 'it', 'ja', 'ko', 'nl', 'pt-br', 'ro', 'ru', 'tr', 'zh-cn']
    with open('top_k/accuracy_results.txt', 'w', encoding='utf-8') as output_file:
        for lang in languages:
            accuracy = process_language(lang)
            output_file.write(f"Language: {lang}, Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    main()
