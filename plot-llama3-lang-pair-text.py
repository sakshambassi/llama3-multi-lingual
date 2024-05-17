import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
from colour import Color

def plot_tsne_embeddings(embeddings, labels, target_words, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the 2D embeddings
    plt.figure(figsize=(12, 12))
    for i, lang in enumerate(set(labels)):
        indices = [j for j, l in enumerate(labels) if l == lang]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=lang)

        # Display tokens for each language
        for j in range(min(5, len(indices))):  # Display tokens for the first 5 words of each language
            plt.text(embeddings_2d[indices[j], 0], embeddings_2d[indices[j], 1], target_words[indices[j]], fontsize=8)

    # Add legend, title, and axis labels
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_path)
    plt.close()

# Specify the directory containing all your CSV files
directory_path = "WordEmbeddingsUpdated/"

# Initialize lists to store data
all_embeddings = []
all_labels = []
all_target_words = []  # New list to store target words

num_colors = 15
start_color = Color("red")
end_color = Color("purple")

# Generate a list of colors between red and purple
clr_list = list(start_color.range_to(end_color, num_colors))

# Iterate through each pair of languages
for lang1 in ['en']:
    for lang2 in ['ar', 'es', 'fr', 'he', 'it', 'ja', 'ko', 'nl', 'pt-br', 'ro', 'ru', 'tr', 'zh-cn']:
        if lang1 == lang2:
            continue

        # Load embeddings and labels for both languages
        embeddings_lang1 = pd.read_csv(os.path.join(directory_path, f"{lang1}_target_word_embeddings.csv")).drop('target_word', axis=1).to_numpy()
        embeddings_lang2 = pd.read_csv(os.path.join(directory_path, f"{lang2}_target_word_embeddings.csv")).drop('target_word', axis=1).to_numpy()
        labels_lang1 = [f"{lang1}_{i}" for i in range(len(embeddings_lang1))]
        labels_lang2 = [f"{lang2}_{i}" for i in range(len(embeddings_lang2))]
        target_words_lang1 = pd.read_csv(os.path.join(directory_path, f"{lang1}_target_word_embeddings.csv"))['target_word'].tolist()
        target_words_lang2 = pd.read_csv(os.path.join(directory_path, f"{lang2}_target_word_embeddings.csv"))['target_word'].tolist()

        # Concatenate embeddings, labels, and target words for both languages
        all_embeddings.append(np.concatenate([embeddings_lang1, embeddings_lang2], axis=0))
        all_labels.extend(labels_lang1 + labels_lang2)
        all_target_words.extend(target_words_lang1 + target_words_lang2)

        # Plot t-SNE embeddings for the current language pair
        plot_tsne_embeddings(np.concatenate([embeddings_lang1, embeddings_lang2], axis=0),
                             labels_lang1 + labels_lang2,
                             target_words_lang1 + target_words_lang2,
                             f"t-SNE Projection of Word Embeddings for {lang1}-{lang2} Languages",
                             f"{lang1}_{lang2}_tsne.png")
