import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sparse

class GloVeModel(nn.Module):
    """GloVe 모델 구현
    
    Args:
        vocab_size: 어휘 사전 크기
        embedding_dim: 임베딩 차원
    """
    def __init__(self, vocab_size, embedding_dim):
        super(GloVeModel, self).__init__()
        
        # 중심 단어 임베딩 행렬
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 문맥 단어 임베딩 행렬
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 중심 단어 편향
        self.center_biases = nn.Embedding(vocab_size, 1)
        # 문맥 단어 편향
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # 가중치 초기화
        self.init_weights()
    
    def init_weights(self):
        """가중치 초기화 함수"""
        # 모든 임베딩 행렬의 가중치를 -0.5 ~ 0.5 사이의 균등 분포로 초기화
        nn.init.uniform_(self.center_embeddings.weight, -0.5, 0.5)
        nn.init.uniform_(self.context_embeddings.weight, -0.5, 0.5)
        
        # 편향 항을 0으로 초기화
        nn.init.zeros_(self.center_biases.weight)
        nn.init.zeros_(self.context_biases.weight)
    
    def forward(self, center_word_idx, context_word_idx):
        """순전파 함수
        
        Args:
            center_word_idx: 중심 단어 인덱스
            context_word_idx: 문맥 단어 인덱스
            
        Returns:
            예측값 (중심 벡터와 문맥 벡터의 내적 + 편향)
        """
        # 임베딩 가져오기
        center_embeds = self.center_embeddings(center_word_idx)  # [batch_size, embedding_dim]
        context_embeds = self.context_embeddings(context_word_idx)  # [batch_size, embedding_dim]
        center_biases = self.center_biases(center_word_idx).squeeze()  # [batch_size]
        context_biases = self.context_biases(context_word_idx).squeeze()  # [batch_size]
        
        # 예측값 계산
        dot_product = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        log_cooccurrence = dot_product + center_biases + context_biases  # [batch_size]
        
        return log_cooccurrence
    
    def get_center_embeddings(self):
        """학습된 중심 단어 임베딩 반환"""
        return self.center_embeddings.weight.detach().cpu().numpy()
    
    def get_context_embeddings(self):
        """학습된 문맥 단어 임베딩 반환"""
        return self.context_embeddings.weight.detach().cpu().numpy()
    
    def get_combined_embeddings(self):
        """중심 단어 임베딩과 문맥 단어 임베딩의 평균 반환"""
        center = self.center_embeddings.weight.detach().cpu().numpy()
        context = self.context_embeddings.weight.detach().cpu().numpy()
        return (center + context) / 2.0

class GloVeDataset(Dataset):
    """GloVe 학습을 위한 데이터셋
    
    동시출현 행렬에서 비영(non-zero) 원소를 추출하여 데이터셋 구성
    
    Args:
        cooccurrence_matrix: 희소 행렬 형태의 동시출현 행렬
        device: 텐서를 저장할 장치 (CPU 또는 GPU)
    """
    def __init__(self, cooccurrence_matrix, device='cpu'):
        self.device = device
        
        # 희소 행렬에서 비영(non-zero) 원소 추출
        self.i_indices, self.j_indices = cooccurrence_matrix.nonzero()
        self.values = cooccurrence_matrix.data
        
        print(f"데이터셋 크기: {len(self.values)}")
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        center_word_idx = self.i_indices[idx]
        context_word_idx = self.j_indices[idx]
        cooccurrence = self.values[idx]
        
        # 텐서로 변환
        center_word_idx = torch.tensor(center_word_idx, dtype=torch.long).to(self.device)
        context_word_idx = torch.tensor(context_word_idx, dtype=torch.long).to(self.device)
        cooccurrence = torch.tensor(cooccurrence, dtype=torch.float).to(self.device)
        log_cooccurrence = torch.log(cooccurrence + 1e-8)  # 수치 안정성을 위해 작은 값 추가
        
        return center_word_idx, context_word_idx, cooccurrence, log_cooccurrence

def glove_loss(predicted, log_cooccurrence, cooccurrence, x_max=100.0, alpha=0.75):
    """GloVe 손실 함수
    
    Args:
        predicted: 모델의 예측값 (w_i^T w_j + b_i + b_j)
        log_cooccurrence: 실제 동시출현 빈도의 로그값 (log X_ij)
        cooccurrence: 실제 동시출현 빈도 (X_ij)
        x_max: 최대 동시출현 빈도 기준값
        alpha: 가중치 함수의 지수
        
    Returns:
        손실 값
    """
    # 가중치 계산
    weights = (cooccurrence / x_max)**alpha
    weights = torch.min(weights, torch.ones_like(weights))
    
    # 손실 계산: f(X_ij) * (w_i^T w_j + b_i + b_j - log X_ij)^2
    squared_diff = torch.pow(predicted - log_cooccurrence, 2)
    weighted_squared_diff = weights * squared_diff
    
    return torch.mean(weighted_squared_diff)

def build_cooccurrence_matrix_glove(tokenized_corpus, word_to_id, window_size=1):
    """GloVe 모델을 위한 동시출현 행렬을 구축하는 함수
    
    Args:
        tokenized_corpus: 토큰화된 말뭉치
        word_to_id: 단어-ID 매핑 딕셔너리
        window_size: 중심 단어 좌우의 윈도우 크기
        
    Returns:
        희소 행렬 형태의 동시출현 행렬
    """
    vocab_size = len(word_to_id)
    cooccurrence_dict = defaultdict(float)
    
    # 모든 문장에 대해 반복
    for sentence in tokenized_corpus:
        # 문장의 길이
        sentence_length = len(sentence)
        
        # 문장의 각 위치에 대해 반복
        for i, center_word in enumerate(sentence):
            # 중심 단어가 어휘 사전에 없는 경우 건너뜀
            if center_word not in word_to_id:
                continue
                
            center_id = word_to_id[center_word]
            
            # 윈도우 내의 단어들에 대해 반복
            window_start = max(0, i - window_size)
            window_end = min(sentence_length, i + window_size + 1)
            
            for j in range(window_start, window_end):
                # 중심 단어 자신은 건너뜀
                if i == j:
                    continue
                    
                context_word = sentence[j]
                
                # 문맥 단어가 어휘 사전에 없는 경우 건너뜀
                if context_word not in word_to_id:
                    continue
                    
                context_id = word_to_id[context_word]
                
                # 거리에 따른 가중치 계산 (거리가 멀수록 낮은 가중치)
                distance = abs(j - i)
                weight = 1.0 / distance
                
                # 동시출현 빈도 증가 (가중치 적용)
                cooccurrence_dict[(center_id, context_id)] += weight
    
    # 희소 행렬 생성을 위한 데이터 준비
    row_indices = []
    col_indices = []
    data = []
    
    for (i, j), value in cooccurrence_dict.items():
        row_indices.append(i)
        col_indices.append(j)
        data.append(value)
    
    # CSR 형식의 희소 행렬 생성
    cooccurrence_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), 
                                            shape=(vocab_size, vocab_size))
    
    return cooccurrence_matrix

class GloVeDataset(Dataset):
    """GloVe 학습을 위한 데이터셋
    
    동시출현 행렬에서 비영(non-zero) 원소를 추출하여 데이터셋 구성
    
    Args:
        cooccurrence_matrix: 희소 행렬 형태의 동시출현 행렬
        device: 텐서를 저장할 장치 (CPU 또는 GPU)
    """
    def __init__(self, cooccurrence_matrix, device='cpu'):
        self.device = device
        
        # 희소 행렬에서 비영(non-zero) 원소 추출
        self.i_indices, self.j_indices = cooccurrence_matrix.nonzero()
        self.values = cooccurrence_matrix.data
        
        print(f"데이터셋 크기: {len(self.values)}")
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        center_word_idx = self.i_indices[idx]
        context_word_idx = self.j_indices[idx]
        cooccurrence = self.values[idx]
        
        # 텐서로 변환
        center_word_idx = torch.tensor(center_word_idx, dtype=torch.long).to(self.device)
        context_word_idx = torch.tensor(context_word_idx, dtype=torch.long).to(self.device)
        cooccurrence = torch.tensor(cooccurrence, dtype=torch.float).to(self.device)
        log_cooccurrence = torch.log(cooccurrence + 1e-8)  # 수치 안정성을 위해 작은 값 추가
        
        return center_word_idx, context_word_idx, cooccurrence, log_cooccurrence
