import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sparse
from collections import defaultdict
class GloVeModel(nn.Module):
    """GloVe model implementation
    
    Args:
        vocab_size: vocabulary size
        embedding_dim: embedding dimension
    """
    def __init__(self, vocab_size, embedding_dim):
        super(GloVeModel, self).__init__()
        
        # Center word embedding matrix
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Context word embedding matrix
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Center word bias
        self.center_biases = nn.Embedding(vocab_size, 1)
        # Context word bias
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # Weight initialization
        self.init_weights()
    
    def init_weights(self):
        """Weight initialization function"""
        # Initialize the weights of all embedding matrices with a uniform distribution between -0.5 and 0.5
        nn.init.uniform_(self.center_embeddings.weight, -0.5, 0.5)
        nn.init.uniform_(self.context_embeddings.weight, -0.5, 0.5)
        
        # Initialize bias terms to 0
        nn.init.zeros_(self.center_biases.weight)
        nn.init.zeros_(self.context_biases.weight)

    def get_text_embedding(self, text, word_to_id):
        """
        Calculate the GloVe embedding for a given text.

        Args:
            text: The input text (string).
            word_to_id: A dictionary mapping words to their IDs.

        Returns:
            A numpy array representing the GloVe embedding for the text.
        """
        # Preprocess the text
        from arxivDataPreProcess import preprocess_text
        processed_text = preprocess_text(text)

        # Convert words to IDs
        word_ids = [word_to_id[word] for word in processed_text if word in word_to_id]

        # If no words are found in the vocabulary, return a zero vector
        if not word_ids:
            return torch.zeros(self.center_embeddings.embedding_dim)

        # Get embeddings
        word_embeddings = self.center_embeddings(torch.tensor(word_ids, dtype=torch.long))

        # Calculate the average embedding
        text_embedding = torch.mean(word_embeddings, dim=0)

        return text_embedding.detach().cpu().numpy()
    
    def forward(self, center_word_idx, context_word_idx):
        """Forward propagation function
        
        Args:
            center_word_idx: center word index
            context_word_idx: context word index
            
        Returns:
            Prediction (dot product of center vector and context vector + bias)
        """
        # Get embedding
        center_embeds = self.center_embeddings(center_word_idx)  # [batch_size, embedding_dim]
        context_embeds = self.context_embeddings(context_word_idx)  # [batch_size, embedding_dim]
        center_biases = self.center_biases(center_word_idx).squeeze()  # [batch_size]
        context_biases = self.context_biases(context_word_idx).squeeze()  # [batch_size]
        
        # Calculate prediction
        dot_product = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        log_cooccurrence = dot_product + center_biases + context_biases  # [batch_size]
        
        return log_cooccurrence
    
    def get_center_embeddings(self):
        """Return learned center word embedding"""
        return self.center_embeddings.weight.detach().cpu().numpy()
    
    def get_context_embeddings(self):
        """Return learned context word embedding"""
        return self.context_embeddings.weight.detach().cpu().numpy()
    
    def get_combined_embeddings(self):
        """Return the average of center word embedding and context word embedding"""
        center = self.center_embeddings.weight.detach().cpu().numpy()
        context = self.context_embeddings.weight.detach().cpu().numpy()
        return (center + context) / 2.0

class GloVeDataset(Dataset):
    """Dataset for GloVe training
    
    Construct a dataset by extracting non-zero elements from the co-occurrence matrix
    
    Args:
        cooccurrence_matrix: co-occurrence matrix in sparse matrix form
        device: device to store tensors (CPU or GPU)
    """
    def __init__(self, cooccurrence_matrix, device='cpu'):
        self.device = device
        
        # Extract non-zero elements from the sparse matrix
        self.i_indices, self.j_indices = cooccurrence_matrix.nonzero()
        self.values = cooccurrence_matrix.data
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        center_word_idx = self.i_indices[idx]
        context_word_idx = self.j_indices[idx]
        cooccurrence = self.values[idx]
        
        # Convert to tensor
        center_word_idx = torch.tensor(center_word_idx, dtype=torch.long).to(self.device)
        context_word_idx = torch.tensor(context_word_idx, dtype=torch.long).to(self.device)
        cooccurrence = torch.tensor(cooccurrence, dtype=torch.float).to(self.device)
        log_cooccurrence = torch.log(cooccurrence + 1e-8)  # Add a small value for numerical stability
        
        return center_word_idx, context_word_idx, cooccurrence, log_cooccurrence

def GloVeLoss(predicted, log_cooccurrence, cooccurrence, x_max=100.0, alpha=0.75):
    """GloVe loss function
    
    Args:
        predicted: predicted value of the model (w_i^T w_j + b_i + b_j)
        log_cooccurrence: log value of the actual co-occurrence frequency (log X_ij)
        cooccurrence: actual co-occurrence frequency (X_ij)
        x_max: maximum co-occurrence frequency reference value
        alpha: exponent of the weight function
        
    Returns:
        Loss value
    """
    # Weight calculation
    weights = (cooccurrence / x_max)**alpha
    weights = torch.min(weights, torch.ones_like(weights))
    
    # Loss calculation: f(X_ij) * (w_i^T w_j + b_i + b_j - log X_ij)^2
    squared_diff = torch.pow(predicted - log_cooccurrence, 2)
    weighted_squared_diff = weights * squared_diff
    
    return torch.mean(weighted_squared_diff)

def build_cooccurrence_matrix_glove(tokenized_corpus, word_to_id, window_size=1):
    """Function to build a co-occurrence matrix for the GloVe model
    
    Args:
        tokenized_corpus: tokenized corpus
        word_to_id: word-ID mapping dictionary
        window_size: window size to the left and right of the center word
        
    Returns:
        Co-occurrence matrix in sparse matrix form
    """
    vocab_size = len(word_to_id)
    cooccurrence_dict = defaultdict(float)
    
    # Iterate over all sentences
    for sentence in tokenized_corpus:
        # Length of the sentence
        sentence_length = len(sentence)
        
        # Iterate over each position in the sentence
        for i, center_word in enumerate(sentence):
            # Skip if the center word is not in the vocabulary
            if center_word not in word_to_id:
                continue
                
            center_id = word_to_id[center_word]
            
            # Iterate over the words in the window
            window_start = max(0, i - window_size)
            window_end = min(sentence_length, i + window_size + 1)
            
            for j in range(window_start, window_end):
                # Skip the center word itself
                if i == j:
                    continue
                    
                context_word = sentence[j]
                
                # Skip if the context word is not in the vocabulary
                if context_word not in word_to_id:
                    continue
                    
                context_id = word_to_id[context_word]
                
                # Calculate the weight according to the distance (lower weight for longer distances)
                distance = abs(j - i)
                weight = 1.0 / distance
                
                # Increase co-occurrence frequency (apply weight)
                cooccurrence_dict[(center_id, context_id)] += weight
    
    print(f"Cooccurrence dictionary: {cooccurrence_dict}")
    # Prepare data for sparse matrix creation
    row_indices = []
    col_indices = []
    data = []
    
    for (i, j), value in cooccurrence_dict.items():
        row_indices.append(i)
        col_indices.append(j)
        data.append(value)
    
    # Create sparse matrix in CSR format
    cooccurrence_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), 
                                            shape=(vocab_size, vocab_size))
    
    return cooccurrence_matrix
