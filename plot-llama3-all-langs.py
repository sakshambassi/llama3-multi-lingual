import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
from colour import Color

# Specify the directory containing all your CSV files
directory_path = 'WordEmbeddingsUpdated'

# Initialize lists to store data
all_embeddings = []
all_labels = []

num_colors = 15
start_color = Color("red")
end_color = Color("blue")

# Generate a list of colors between red and purple
clr_list = list(start_color.range_to(end_color, num_colors))

# Iterate through each CSV file in the directory
for color, file_name in zip(clr_list, os.listdir(directory_path)):
    if file_name.endswith('.csv'):
        # Load the CSV file with word embeddings
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)

        # Extract word embeddings
        embeddings = df.drop('target_word', axis=1).to_numpy()

        # Append embeddings and labels to the lists
        all_embeddings.append(embeddings)
        all_labels.extend([f"{file_name[:2]}_{i}" for i in range(len(embeddings))])

# Concatenate all embeddings and perform t-SNE
all_embeddings = np.concatenate(all_embeddings, axis=0)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot the 2D embeddings with different colors for each language
plt.figure(figsize=(12, 12))
for color, label in zip(clr_list, os.listdir(directory_path)):
    indices = [i for i, l in enumerate(all_labels) if l.startswith(label[:2])]
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label[:2], color=color.hex)
    
    # Display tokens for each language
    for i in range(min(5, len(indices))):  # Display tokens for the first 5 words of each language
        plt.text(embeddings_2d[indices[i], 0], embeddings_2d[indices[i], 1], all_labels[indices[i]], fontsize=8)

# Add legend, title, and axis labels
plt.legend()
plt.title("t-SNE Projection of Word Embeddings for Multiple Languages")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig("tsne.png")
