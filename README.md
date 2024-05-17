# llama3-multi-lingual
Analysis of LlaMa3 on multiple languages

## Setup singularity cluster with miniconda
Some necessary libraries like:
```
torch-gpu
transformers
sentencepiece
scikit-learn
```
View [this link](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda) if running on NYU Greene.


## Details about the code
1. `llama3-final-layer.py`: This is the starting code to generate the embeddings from the last layer of llama3 for each token provided in the input directory. Make sure you have the `Languages` folder with all the CSVs and create an empty folder called `WordEmbeddingsUpdated` (or anything else but make sure to change it in code).
2. `llama3-alternate-layers-embeddings.py`: This is currently a Work in Progress, as the name suggests it does the above for all alternate layers (i.e. to analyze output from hidden layers)
3. `plot-llama3-all-langs.py`: This plots the embeddings in the TSNE plot for all the languages. Outputs can be found in the `tsne` folder.
4. `plot-llama3-all-langs-text.py` and `plot-llama3-lang-pair-text.py`: Current Work In Progress, but they add text over in the TSNE plot, and the pairing one only output for `en` and one other language. Outputs are in the `tsne` folder.
5. `generate-top-k-keywords-en.py`: This code generates the top-k keywords in a given language (low-res) for each word in English language. The output files are in `top_k` folder.
6. `analysis-top-k-keywords.py`: This code calculates the accuracy of the top-k keywords i.e. was the direct translated keyword (english to another language) included in top-k list or not. Output is in `top_k/accuracy_results.txt`
