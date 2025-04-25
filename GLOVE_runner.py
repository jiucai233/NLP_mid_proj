import time
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from GLOVE_model import GloVeModel, glove_loss, GloVeDataset
import arxiv
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import scipy.sparse as sparse
from collections import defaultdict

nltk.download('punkt_tab')
nltk.download('stopwords')

# 数据加载和预处理
def load_and_preprocess_data():
    # 从 arxiv 下载数据
    search = arxiv.Search(
        query="cs.AI",
        max_results=100,  # 增加结果数量
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    data = []
    for result in search.results():
        authors = ", ".join([author.name for author in result.authors])
        data.append([result.title, authors, result.summary.replace('\n', ' '), str(result.published)])
    df = pd.DataFrame(data, columns=['Title', 'Authors', 'Abstract', 'Date'])

    # 文本预处理
    def preprocess_text(text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if token not in string.punctuation]
        return tokens

    # 对所有 abstract 进行预处理
    all_abstracts = df['Abstract'].tolist()
    processed_abstracts = [preprocess_text(abstract) for abstract in all_abstracts]
    return processed_abstracts

# 构建词汇表
def build_vocabulary(processed_abstracts):
    word_counts = Counter()
    for abstract in processed_abstracts:
        word_counts.update(abstract)
    word_to_id = {word: i for i, word in enumerate(word_counts.keys())}
    return word_to_id

# GloVe 모델을 위한 동시출현 행렬을 구축하는 함수
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

# 加载和预处理数据
processed_abstracts = load_and_preprocess_data()

# 构建词汇表
word_to_id = build_vocabulary(processed_abstracts)
vocab_size = len(word_to_id)

# 构建共现矩阵
tokenized_corpus = processed_abstracts  # tokenized_corpus를 processed_abstracts로 설정
cooccurrence_matrix = build_cooccurrence_matrix_glove(tokenized_corpus, word_to_id)

# GloVe 데이터셋 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove_dataset = GloVeDataset(cooccurrence_matrix, device=device)

# 데이터 로더 생성
batch_size = 1024
data_loader = DataLoader(glove_dataset, batch_size=batch_size, shuffle=True)

# 하이퍼파라미터 설정
embedding_dim = 50  # 임베딩 차원
learning_rate = 0.05  # 학습률
num_epochs = 30  # 학습 에폭 수
x_max = 100.0  # 최대 동시출현 빈도 기준값
alpha = 0.75  # 가중치 함수의 지수

# 모델 초기화
model = GloVeModel(vocab_size, embedding_dim)

# 옵티마이저 설정
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

# 손실 기록을 위한 리스트
losses = []

# 학습 시작 시간 기록
start_time = time.time()

# 학습 루프
for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    for center_word_idx, context_word_idx, cooccurrence, log_cooccurrence in data_loader:
        # 순전파
        predicted = model(center_word_idx, context_word_idx)
        
        # 손실 계산
        loss = glove_loss(predicted, log_cooccurrence, cooccurrence, x_max, alpha)
        
        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실 누적
        epoch_loss += loss.item()
        batch_count += 1
    
    # 에폭당 평균 손실 계산
    avg_epoch_loss = epoch_loss / batch_count
    losses.append(avg_epoch_loss)
    
    # 학습 진행 상황 출력
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Time: {elapsed_time:.2f}s")

# 학습 소요 시간 출력
total_time = time.time() - start_time
print(f"\n학습 완료: 총 {num_epochs} 에폭, 소요 시간: {total_time:.2f}초")
