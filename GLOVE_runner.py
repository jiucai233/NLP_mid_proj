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
    
# 数据加载
data_path = "data/arxiv_papers.pkl"
raw_abstract_data = read_abstract_from_pkl(data_path)

# 加载和预处理数据
processed_abstracts = [preprocess_text(abstract) for abstract in raw_abstract_data]

# 构建词汇表
word_to_id, id_to_word, word_counts = build_vocabulary(processed_abstracts)
vocab_size = len(word_to_id)

# 构建共现矩阵
tokenized_corpus = processed_abstracts  # tokenized_corpus를 processed_abstracts로 설정
cooccurrence_matrix = build_cooccurrence_matrix_glove(tokenized_corpus, word_to_id)

# generate GloVe dataset
glove_dataset = GloVeDataset(cooccurrence_matrix)
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
        loss = GloVeLoss(predicted, log_cooccurrence, cooccurrence, x_max, alpha)
        
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

# 학습된 임베딩 가져오기
center_embeddings = model.get_center_embeddings()
context_embeddings = model.get_context_embeddings()
combined_embeddings = model.get_combined_embeddings()


# 임베딩 저장
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
