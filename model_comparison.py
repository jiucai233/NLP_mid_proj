import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pickle
from TFIDF_main import get_tfidf_vectors, find_similar_abstract
from GLOVE_model import GloVeModel
from arxivDataPreProcess import preprocess_text
import torch
import pandas as pd

# 1. TF-IDF Model Evaluation
def evaluate_tfidf(tfidf_matrix, query_vector, true_labels=None, predicted_labels=None):
    """
    Evaluates the TF-IDF model based on cosine similarity and optional classification metrics.
    Returns a dict with similarity scores and evaluation metrics.
    """
    query_vector = query_vector.reshape(1, -1)
    cosine_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results = {'cosine_scores': cosine_scores}

    if true_labels is not None and predicted_labels is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        results.update({'precision': precision, 'recall': recall, 'f1': f1})
        print(f"TF-IDF Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print(f"TF-IDF Cosine Similarities (sample): {np.round(cosine_scores[:5],4)}")

    return results

# 2. GloVe Model Evaluation
def evaluate_glove(model, input_embedding, df, word_to_id):
    """
    Evaluates the GloVe model by computing cosine similarity for each abstract.
    Returns all similarity scores and the best match.
    """
    for abstract in df['Abstract']:
        emb = model.get_text_embedding(abstract, word_to_id)
        abstract_embeddings.append(emb)
    abstract_embeddings = np.vstack(abstract_embeddings)

    similarities = cosine_similarity(input_embedding.reshape(1, -1), abstract_embeddings).flatten()
    best_idx = np.argmax(similarities)
    return {
        'all_scores': similarities,
        'best_title': df['Title'].iloc[best_idx],
        'best_score': similarities[best_idx]
    }

# 3. Model Comparison
def compare_models(tfidf_scores, glove_scores):
    """Compare TF-IDF and GloVe by mean cosine similarity."""
    mean_tfidf = np.mean(tfidf_scores)
    mean_glove = np.mean(glove_scores)
    print(f"Mean TF-IDF Cosine: {mean_tfidf:.4f}")
    print(f"Mean GloVe  Cosine: {mean_glove:.4f}")
    if mean_tfidf > mean_glove:
        print("TF-IDF model outperforms GloVe on average similarity.")
    else:
        print("GloVe model outperforms TF-IDF on average similarity.")

# Visualization helper
def visualize_similarity(similarity_scores, title):
    plt.figure()
    plt.hist(similarity_scores, bins=50)
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    # Load data
    data_path = "data/arxiv_papers.pkl"
    try:
        df = pd.read_pickle(data_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit(1)

    # Example input
    input_text = "attention mechanism"

    # TF-IDF Evaluation
    cosine_similarities, query_vector, best_abstract, vectorizer = get_tfidf_vectors(input_text, df)
    # Pass the vectorizer to find_similar_abstract
    summary, title, metrics = find_similar_abstract(input_text, df, vectorizer)

    tfidf_results = evaluate_tfidf(
        cosine_similarities,
        query_vector,
        true_labels=None,
        predicted_labels=None
    )
    print(f"best_abstract: {best_abstract}")
    visualize_similarity(tfidf_results['cosine_scores'], "TF-IDF Similarity Distribution")

    # Load GloVe embeddings and model
    try:
        with open('glove_embeddings_results.pkl', 'rb') as f:
            data = pickle.load(f)
        embeddings = data['center_embeddings']
        word_to_id = data['word_to_id']
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        exit(1)

    model = GloVeModel(embeddings.shape[0], embeddings.shape[1])
    model.center_embeddings.weight = torch.nn.Parameter(torch.tensor(embeddings))
    model.eval()

    # Prepare input embedding from TF-IDF best abstract
    processed = preprocess_text(best_abstract)
    idxs = [word_to_id[w] for w in processed if w in word_to_id]
    if not idxs:
        print("No known words in input for GloVe.")
        exit(1)
    inp_emb = embeddings[idxs].mean(axis=0)

    glove_results = evaluate_glove(model, inp_emb, df, word_to_id)
    print(f"GloVe Best Title: {glove_results['best_title']}")
    print(f"GloVe Best Score: {glove_results['best_score']:.4f}")
    visualize_similarity(glove_results['all_scores'], "GloVe Similarity Distribution")

    # Compare
    compare_models(tfidf_results['cosine_scores'], glove_results['all_scores'])
