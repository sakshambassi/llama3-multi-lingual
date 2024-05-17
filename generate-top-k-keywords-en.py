import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

def get_top_similar_words(en_words, other_lang_words, top_n=5):
    top_similar_words = {}
    for en_word, en_embedding in en_words.items():
        similarities = []
        for other_word, other_embedding in other_lang_words.items():
            cosine_sim = 1 - cosine(en_embedding, other_embedding)
            similarities.append((other_word, cosine_sim))
        
        # Sort similarities in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top N similar words
        top_n_similar = [word for word, _ in similarities[:top_n]]
        top_similar_words[en_word] = top_n_similar
    
    return top_similar_words

# Load English embeddings
en_df = pd.read_csv('WordEmbeddingsUpdated/en_target_word_embeddings.csv')
en_embeddings = {word: embedding for word, embedding in zip(en_df['target_word'], en_df.drop('target_word', axis=1).to_numpy())}

langs = ['ar', 'en', 'es', 'fr', 'he', 'it', 'ja', 'ko', 'nl', 'pt-br', 'ro', 'ru', 'zh-cn']

for lang in langs:
    # Load other language embeddings
    other_lang_df = pd.read_csv(f'WordEmbeddingsUpdated/{lang}_target_word_embeddings.csv')
    other_lang_embeddings = {word: embedding for word, embedding in zip(other_lang_df['target_word'], other_lang_df.drop('target_word', axis=1).to_numpy())}

    # Get top 5 similar words for each English word
    top_similar = get_top_similar_words(en_embeddings, other_lang_embeddings, top_n=5)

    # Write the results to a text file
    with open(f'top_k/top_similar_words_en_{lang}.txt', 'w', encoding='utf-8') as file:
        for en_word, similar_words in top_similar.items():
            file.write(f"{en_word}: {', '.join(similar_words)}\n")
