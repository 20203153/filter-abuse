from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# 모델과 토크나이저 로드
model_name = "monologg/kobert"  # 또는 "roberta-base" 등
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # num_labels는 레이블 수
raw_dataset = pd.read_csv('./selectstar-location.csv')

def create_labels(sentence, positions):
    tokens = list(sentence)  # 문장을 문자 단위로 토큰화
    labels = ['O'] * len(tokens)  # 초기 레이블 설정

    if not positions:  # positions가 비어있는 경우
        return tokens, labels

    for start, end in positions:
        labels[start] = 'B-PROFANITY'
        for i in range(start + 1, end + 1):
            if i < len(labels):  # 인덱스가 범위를 벗어나지 않도록
                labels[i] = 'I-PROFANITY'

    return tokens, labels

# 데이터셋 생성하기
dataset = []
for index, row in raw_dataset.iterrows():
    tokens, labels = create_labels(row['문장'], row['욕설 위치'])
    dataset.append((tokens, labels))

texts, labels = zip(*dataset)

X_train, X_eval, Y_train, Y_eval = train_test_split(texts, labels, test_size=0.2)

# Tokenization 및 인덱싱
train_encodings = tokenizer(list(X_train), is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
eval_encodings = tokenizer(list(X_eval), is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")

# 레이블을 인덱스로 변환
label_to_index = {'O': 0, 'B-PROFANITY': 1, 'I-PROFANITY': 2}
train_labels = [[label_to_index[label] for label in label_list] for label_list in Y_train]
eval_labels = [[label_to_index[label] for label in label_list] for label_list in Y_eval]

class ProfanityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
# 데이터셋 생성
train_dataset = ProfanityDataset(train_encodings, train_labels)
eval_dataset = ProfanityDataset(eval_encodings, eval_labels)

# Trainer 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()