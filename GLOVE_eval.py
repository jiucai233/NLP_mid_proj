import torch
import pickle
from arxivDataPreProcess import read_abstract_from_pkl, build_vocabulary, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(embedding1, embedding2):
    """Calculates the cosine similarity between two embeddings."""
    dot_product = torch.dot(embedding1, embedding2)
    norm_embedding1 = torch.norm(embedding1)
    norm_embedding2 = torch.norm(embedding2)
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    return similarity.item()


def get_phrase_embedding(phrase, word_to_id, combined_embeddings):
    """Calculates the average GloVe embedding for a given phrase."""
    # Collect embeddings of valid words
    word_embeddings = [torch.tensor(combined_embeddings[word_to_id[token]]) for token in phrase if token in word_to_id]

    if not word_embeddings:
        return None  # Return None if no valid word is found

    # Calculate average embedding
    phrase_embedding = torch.mean(torch.stack(word_embeddings), dim=0)
    return phrase_embedding


def extract_keywords_tfidf(text, top_n=5):
    """Extracts the top N keywords from the given text using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Sort keywords by TF-IDF score
    ranked_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

    # Extract top N keywords
    keywords = [keyword for keyword, score in ranked_keywords[:top_n]]
    return keywords


def recommend_papers(phrase, word_to_id, combined_embeddings, raw_abstract_data):
    """Recommends papers based on the cosine similarity between the phrase and paper abstracts."""
    phrase_embedding = get_phrase_embedding(phrase, word_to_id, combined_embeddings)

    if phrase_embedding is None:
        print("No valid words found in the phrase. Cannot generate recommendations.")
        return []

    similarities = []
    for abstract in raw_abstract_data:
        # abstract_text = " ".join(abstract)  # 确保 abstract 是一个字符串
        # print(f"Abstract text in recommend_papers: {abstract}")  # 添加打印语句
        abstract_keywords = extract_keywords_tfidf(abstract)
        abstract_embedding = get_phrase_embedding(abstract_keywords, word_to_id, combined_embeddings)
        if abstract_embedding is not None:
            similarity = cosine_similarity(phrase_embedding, abstract_embedding)
            similarities.append(similarity)
        else:
            similarities.append(-1)  # Assign -1 if abstract embedding is None

    # Rank papers based on similarity scores
    ranked_papers = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # Recommend top 5 papers
    top_papers = ranked_papers[:5]

    recommendations = []
    for index, similarity in top_papers:
        recommendations.append({
            'index': index,
            'similarity': similarity,
            'abstract': raw_abstract_data[index]
        })

    return recommendations


def calculate_metrics(phrase, recommendations, word_to_id, combined_embeddings, raw_abstract_data):
    """Calculates evaluation metrics for the recommendations by embedding phrase and abstract together."""

    if not recommendations:
        print("No recommendations found. Cannot calculate metrics.")
        return {}

    metrics = []
    for paper in recommendations:
        abstract = paper['abstract']
        phrase_str = " ".join(phrase)
        # concatenated_text = phrase_str + " " + abstract
        # preprocessed_text = preprocess_text(concatenated_text)
        preprocessed_phrase = preprocess_text(phrase_str)
        preprocessed_abstract = preprocess_text(" ".join(abstract))

        phrase_embedding = get_phrase_embedding(preprocessed_phrase, word_to_id, combined_embeddings)
        # abstract_keywords = extract_keywords_tfidf(preprocessed_abstract)
        abstract_embedding = get_phrase_embedding(preprocessed_abstract, word_to_id, combined_embeddings)

        if phrase_embedding is None or abstract_embedding is None:
            metrics.append(0.0)
            continue

        similarity = cosine_similarity(phrase_embedding, abstract_embedding)
        metrics.append(similarity)

    avg_similarity = sum(metrics) / len(metrics) if metrics else 0.0
    loss = 1 - avg_similarity

    final_metrics = {
        'cosine_similarity': avg_similarity,
        'loss': loss
    }

    return final_metrics

if __name__ == '__main__':
    # Data loading
    data_path = "data/arxiv_papers.pkl"
    raw_abstract_data = read_abstract_from_pkl(data_path)

    # Load embedding results
    with open('glove_embeddings_results.pkl', 'rb') as f:
        embeddings_results = pickle.load(f)

    center_embeddings = embeddings_results['center_embeddings']
    context_embeddings = embeddings_results['context_embeddings']
    combined_embeddings = embeddings_results['combined_embeddings']
    word_to_id = embeddings_results['word_to_id']
    embedding_dim = embeddings_results['embedding_dim']
    losses = embeddings_results['losses']

    import sys

    # Get phrase from command line arguments
    if len(sys.argv) > 1:
        phrase = sys.argv[1]
    else:
        phrase = "natural language processing"  # Default phrase

    phrase = preprocess_text(phrase)
    recommendations = recommend_papers(phrase, word_to_id, combined_embeddings, raw_abstract_data)

    if recommendations:
        print("\nRecommended papers:")
        for paper in recommendations:
            print(f"Index: {paper['index']}, Similarity: {paper['similarity']:.4f}")
            print(f"Abstract: {paper['abstract'][:200]}...\n")
    else:
        print("No recommendations found.")

    metrics = calculate_metrics(phrase, recommendations, word_to_id, combined_embeddings, raw_abstract_data)

    if metrics:
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No metrics calculated.")

    import matplotlib.pyplot as plt

    keywords = [
        "artificial intelligence",
        "machine learning",
        "natural language processing",
        "deep learning",
        "computer vision",
        "data mining",
        "information retrieval",
        "knowledge representation",
        "reasoning",
        "planning",
        "robotics",
        "cognitive science",
        "neural networks",
        "fuzzy logic",
        "expert systems",
        "big data",
        "cloud computing",
        "internet of things",
        "cybersecurity",
        "bioinformatics"
    ]

    all_metrics = []
    for keyword in keywords:
        keyword = preprocess_text(keyword)
        recommendations = recommend_papers(keyword, word_to_id, combined_embeddings, raw_abstract_data)
        metrics = calculate_metrics(keyword, recommendations, word_to_id, combined_embeddings, raw_abstract_data)
        if metrics:
            all_metrics.append(metrics)
            print(f"Keyword: {keyword}")
            print("Metrics:", metrics)
        else:
            print(f"No metrics calculated for keyword: {keyword}")
            all_metrics.append({'cosine_similarity': 0, 'loss': 1})


    # Visualize metric fluctuations (example for cosine_similarity)
    cosine_similarities = [m['cosine_similarity'] for m in all_metrics]
    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
    plt.plot(keywords, cosine_similarities)
    plt.xlabel("Keywords")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity Fluctuation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
