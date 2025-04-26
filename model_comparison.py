import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
from TFIDF_main import read_abstract_from_pkl, find_similar_abstract
from GLOVE_runner import embeddings_results
from GLOVE_model import GloVeModel, GloVeLoss, GloVeDataset, build_cooccurrence_matrix_glove
from arxivDataPreProcess import preprocess_text, build_vocabulary
import torch
import pandas as pd

# 1. TF-IDF Model Evaluation
def evaluate_tfidf(tfidf_matrix, query_vector, true_labels, predicted_labels):
    """
    Evaluates the TF-IDF model based on cosine similarity.
    Args:
        tfidf_matrix: TF-IDF matrix.
        query_vector: Query vector.
        true_labels: True labels for F1 score calculation.
        predicted_labels: Predicted labels for F1 score calculation.
    Returns:
        cosine_sim: Cosine similarity between TF-IDF matrix and query vector.
        f1: F1 score (if applicable).
    """
    cosine_sim = cosine_similarity(tfidf_matrix, query_vector)
    print("TF-IDF Cosine Similarity:\n", cosine_sim)

    # Calculate F1 score (if applicable)
    f1 = None
    #if true_labels is not None and predicted_labels is not None:
    #    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    #    print("TF-IDF F1 Score:", f1)

    # Visualize cosine similarity (example)
    #if cosine_sim is not None:
    #    visualize_similarity(cosine_sim, "TF-IDF Cosine Similarity")

    return cosine_sim, f1

# 2. GloVe Model Evaluation
def evaluate_glove(model, input_embedding, df, word_to_id):
    """
    Evaluates the GloVe model based on cosine similarity.

    Args:
        model: GloVe model.
        input_embedding: Embedding of the input text.
        df: DataFrame containing paper titles and abstracts.
        word_to_id: Word-to-ID mapping.

    Returns:
        title: Title of the most similar paper.
        similarity: Cosine similarity between the input embedding and the most similar abstract embedding.
    """
    # Calculate GloVe embeddings for paper abstracts
    abstract_embeddings = []
    for abstract in df['abstract']:
        abstract_embedding = model.get_text_embedding(abstract, word_to_id)
        abstract_embeddings.append(abstract_embedding)
    abstract_embeddings = np.array(abstract_embeddings)

    # Calculate cosine similarities between input embedding and abstract embeddings
    similarities = cosine_similarity(input_embedding.reshape(1, -1), abstract_embeddings)[0]

    # Find the index of the most similar abstract
    most_similar_index = np.argmax(similarities)

    # Get the title of the most similar paper
    title = df['title'][most_similar_index]
    similarity = similarities[most_similar_index]

    return title, similarity

# 3. Model Comparison
def compare_models(tfidf_cosine_sim, glove_cosine_sim):
    """
    Compares the TF-IDF and GloVe models based on cosine similarity.
    Args:
        tfidf_cosine_sim: Cosine similarity from TF-IDF model.
        glove_cosine_sim: Cosine similarity from GloVe model.
    Returns:
        None
    """
    print("\nModel Comparison:")
    if tfidf_cosine_sim > glove_cosine_sim:
        print("TF-IDF model performed better based on average cosine similarity.")
    else:
        print("GloVe model performed better based on average cosine similarity.")

def visualize_similarity(similarity_scores, title):
    """
    Visualizes the similarity scores using a histogram.
    Args:
        similarity_scores: A numpy array of similarity scores.
        title: The title of the plot.
    Returns:
        None
    """
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
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        df = None
    except Exception as e:
        print(f"Error loading data: {e}")
        df = None

    if df is not None:
        # TF-IDF Evaluation
        input_text = "attention mechanism"  # Replace with actual input text
        most_similar_abstract, title, metrics = find_similar_abstract(input_text, df)
        print("TF-IDF Results:")
        print("Most similar abstract:", most_similar_abstract)
        print("Title:", title)
        print("Metrics:", metrics)

        # Load GloVe embeddings
        try:
            with open('glove_embeddings_results.pkl', 'rb') as f:
                embeddings_results = pickle.load(f)
            center_embeddings = embeddings_results['center_embeddings']
            word_to_id = embeddings_results['word_to_id']
            print("GloVe embeddings loaded successfully.")
        except FileNotFoundError:
            print("Error: glove_embeddings_results.pkl not found.")
            center_embeddings = None
            word_to_id = None
        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            center_embeddings = None
            word_to_id = None

        if center_embeddings is not None and word_to_id is not None:
            # Load GloVe model
            model = GloVeModel(center_embeddings.shape[0], center_embeddings.shape[1])
            model.center_embeddings.weight = torch.nn.Parameter(torch.tensor(center_embeddings))
            model.eval()

            # Prepare GloVe data
            #processed_text = preprocess_text(input_text)
            processed_text = preprocess_text(most_similar_abstract)

            # Convert words to IDs
            input_ids = [word_to_id[word] for word in processed_text if word in word_to_id]

            # If no words are found in the vocabulary, print a message and exit
            if not input_ids:
                print("No words from the input text found in the vocabulary.")
            else:
                # Convert to tensor
                input_tensor = torch.tensor(input_ids, dtype=torch.long)

                # Get embeddings
                input_embedding = center_embeddings[input_tensor].mean().numpy()

                # Evaluate GloVe model
                glove_title, glove_similarity = evaluate_glove(model, input_embedding, df, word_to_id)
                print("\nGloVe Results:")
                print("Most similar title:", glove_title)
                print("Cosine Similarity:", glove_similarity)

            # Compare models
            compare_models(metrics['cosine_similarity'], glove_similarity)
        else:
            print("Skipping GloVe evaluation due to missing embeddings or word-to-id mapping.")
    else:
        print("Skipping evaluation due to missing data.")
