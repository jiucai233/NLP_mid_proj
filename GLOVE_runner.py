from GLOVE_model import GloVeModel, GloVeLoss, GloVeDataset, build_cooccurrence_matrix_glove
from arxivDataPreProcess import read_abstract_from_pkl, build_vocabulary, preprocess_text
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import nltk
import pickle

nltk.download('punkt_tab')
nltk.download('stopwords')
    
# Data loading
data_path = "data/arxiv_papers.pkl"
raw_abstract_data = read_abstract_from_pkl(data_path)

# Load and preprocess data
processed_abstracts = [preprocess_text(abstract) for abstract in raw_abstract_data]

# Build vocabulary
word_to_id, id_to_word, word_counts = build_vocabulary(processed_abstracts)
vocab_size = len(word_to_id)

# Build co-occurrence matrix
tokenized_corpus = processed_abstracts  # set tokenized_corpus to processed_abstracts
cooccurrence_matrix = build_cooccurrence_matrix_glove(tokenized_corpus, word_to_id)

# generate GloVe dataset
glove_dataset = GloVeDataset(cooccurrence_matrix)
# Create data loader
batch_size = 1024
data_loader = DataLoader(glove_dataset, batch_size=batch_size, shuffle=True)

# Set hyperparameters
embedding_dim = 50  # Embedding dimension
learning_rate = 0.05  # Learning rate
num_epochs = 30  # Number of training epochs
x_max = 100.0  # Maximum co-occurrence frequency reference value
alpha = 0.75  # Exponent of the weight function

# Initialize model
model = GloVeModel(vocab_size, embedding_dim)

# Set optimizer
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

# List for recording loss
losses = []

# Record training start time
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    for center_word_idx, context_word_idx, cooccurrence, log_cooccurrence in data_loader:
        # Forward propagation
        predicted = model(center_word_idx, context_word_idx)
        
        # Loss calculation
        loss = GloVeLoss(predicted, log_cooccurrence, cooccurrence, x_max, alpha)
        
        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss accumulation
        epoch_loss += loss.item()
        batch_count += 1
    
    # Calculate average loss per epoch
    avg_epoch_loss = epoch_loss / batch_count
    losses.append(avg_epoch_loss)
    
    # Print training progress
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Time: {elapsed_time:.2f}s")

# Print training time
total_time = time.time() - start_time
print(f"\n학습 완료: 총 {num_epochs} 에폭, 소요 시간: {total_time:.2f}초")

# Get trained embedding
center_embeddings = model.get_center_embeddings()
context_embeddings = model.get_context_embeddings()
combined_embeddings = model.get_combined_embeddings()


# Save embedding
embeddings_results = {
    'center_embeddings': center_embeddings,
    'context_embeddings': context_embeddings,
    'combined_embeddings': combined_embeddings,
    'word_to_id': word_to_id,
    'embedding_dim': embedding_dim,
    'losses': losses
}

with open('glove_embeddings_results.pkl', 'wb') as f:
    pickle.dump(embeddings_results, f)

print("GloVe 임베딩 결과가 'glove_embeddings_results.pkl' 파일에 저장되었습니다.")
