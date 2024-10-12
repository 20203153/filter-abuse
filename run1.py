import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# Custom dataset class
class OffensiveLanguageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = 256):
        self.texts = dataframe['문장'].tolist()
        self.start_labels = []
        self.end_labels = []
        self.targets = dataframe['욕설'].tolist()  # 전체 욕설 여부
        self.tokenizer = tokenizer
        self.max_len = max_len

        for index, row in dataframe.iterrows():
            # 욕설 위치를 리스트로 파싱: 실제 tuple인지확인
            spans = row['욕설 위치']
            if isinstance(spans, list) and len(spans) > 0:  # 리스트가 비어있지 않다면
                try:
                    self.start_labels.append([start for start, end in spans])  # (start, end) 튜플로 분리
                    self.end_labels.append([end for start, end in spans])
                except ValueError:  # unpacking mistake
                    print(f"Error unpacking spans in row {index}: {spans}")
                    self.start_labels.append([])
                    self.end_labels.append([])
            else:
                self.start_labels.append([])
                self.end_labels.append([])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        start_label = self.start_labels[index]
        end_label = self.end_labels[index]
        target = self.targets[index]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Prepare the return dictionary
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_labels': torch.tensor(start_label, dtype=torch.float),  # 장면별 욕설 시작 인덱스
            'end_labels': torch.tensor(end_label, dtype=torch.float),      # 장면별 욕설 끝 인덱스
            'targets': torch.tensor(target, dtype=torch.float)              # 전체 욕설 여부
        }

class MultiOffensiveLanguageModel(nn.Module):
    def __init__(self, model_name, num_labels=1):
        super(MultiOffensiveLanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classification_layer = nn.Linear(self.bert.config.hidden_size, num_labels)  # 욕설 여부
        self.start_layer = nn.Linear(self.bert.config.hidden_size, 1)  # 시작 위치
        self.end_layer = nn.Linear(self.bert.config.hidden_size, 1)  # 끝 위치
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        classification_logits = self.classification_layer(hidden_states)  # [batch_size, seq_len, num_labels]
        start_logits = self.start_layer(hidden_states)  # [batch_size, seq_len, 1]
        end_logits = self.end_layer(hidden_states)  # [batch_size, seq_len, 1]
        
        return classification_logits, start_logits, end_logits

# 하이퍼파라미터 설정
max_len = 256
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# Tokenizer 및 모델 초기화
model_name = 'monologg/kobert'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = MultiOffensiveLanguageModel(model_name)

# 데이터셋 및 데이터로더 준비
df = pd.read_csv('selectstar-location.csv')

dataset = OffensiveLanguageDataset(df, tokenizer, max_len)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer 및 Loss Function 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_function_cls = nn.BCEWithLogitsLoss()  # 욕설 여부에 대한 손실 함수
loss_function_loc = nn.MSELoss()  # 시작 및 끝 위치에 대한 손실 함수

# Training loop
num_epochs = 5
model.train()

for epoch in tqdm(range(num_epochs), "Epoch", position=1):
    for batch in tqdm(train_loader, "Batch", position=0):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_labels = batch['start_labels']
        end_labels = batch['end_labels']
        targets = batch['targets']

        optimizer.zero_grad()
        
        # Forward pass
        classification_logits, start_logits, end_logits = model(input_ids, attention_mask)

        # Average pooling for classification logits
        classification_logits = classification_logits.mean(dim=1).view(-1, 1)  # Shape to (batch_size, 1)

        # Calculate loss for classification
        loss_cls = loss_function_cls(classification_logits, targets.view(-1, 1))

        # Reshape logits for start and end positions
        start_logits = start_logits.view(-1, 1)  # Shape to (batch_size * seq_len, 1)
        end_logits = end_logits.view(-1, 1)      # Shape to (batch_size * seq_len, 1)

        # Ensure labels have compatible shapes
        if start_labels.nelement() > 0:  # Check if start_labels is not empty
            start_labels = start_labels.view(-1, 1)  # Ensure dimension is (batch_size * max_seq_len, 1)
            loss_start = loss_function_loc(start_logits, start_labels)
        else:
            loss_start = torch.tensor(0.0, requires_grad=True)  # No contribution to the loss if empty

        if end_labels.nelement() > 0:  # Check if end_labels is not empty
            end_labels = end_labels.view(-1, 1)
            loss_end = loss_function_loc(end_logits, end_labels)
        else:
            loss_end = torch.tensor(0.0, requires_grad=True)

        loss = loss_cls + loss_start + loss_end
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
# 모델 학습이 완료되었습니다.

# 모델 저장
model_save_path = './my_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f'Model saved at {model_save_path}')