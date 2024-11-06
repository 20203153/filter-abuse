from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, hamming_loss
import os
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

# 레이블 설정 (영어로 의미 표현)
label_categories = [
    "INSULT",  # 모욕
    "PROFANE",  # 욕설
    "OBSCENE",  # 외설
    "THREAT",  # 폭력위협/범죄조장
    "HATE_SPEECH",  # 성혐오
    "AGEISM",  # 연령
    "RACISM",  # 인종/지역
    "DISABILITY",  # 장애
    "RELIGION",  # 종교
    "POLITICAL",  # 정치성향
    "PROFESSION"  # 직업
]

# label2id 및 id2label 정의
label2id = {label: i for i, label in enumerate(label_categories)}
id2label = {i: label for i, label in enumerate(label_categories)}

def load_and_preprocess_data():
    # 데이터 로드
    raw_dataset = pd.read_csv('./selectstar-encoded.csv')

    # 레이블 데이터 생성
    raw_dataset['labels'] = raw_dataset[label_categories].values.tolist()
    
    # 데이터 분할
    X_train, X_eval, Y_train, Y_eval = train_test_split(
        raw_dataset['text'].tolist(), 
        raw_dataset['labels'].tolist(), 
        test_size=0.2,
        stratify=raw_dataset['labels'].apply(lambda x: tuple(x))  # 튜플로 변환하여 stratified 분할
    )
    
    return X_train, X_eval, Y_train, Y_eval

if __name__ == "__main__":
    # CUDA, MPS, CPU 체크
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/KcELECTRA-base",
        num_labels=len(label_categories),  # 레이블 수
        id2label=id2label,  # id2label 설정
        label2id=label2id,  # label2id 설정
        problem_type="multi_label_classification"
    ).to(device)
    
    # 데이터셋 로드 및 변환
    X_train, X_eval, Y_train, Y_eval = load_and_preprocess_data()

    # Tokenization 및 인덱싱
    train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt", max_length=256)
    eval_encodings = tokenizer(X_eval, padding=True, truncation=True, return_tensors="pt", max_length=256)

    # PyTorch Dataset 정의
    class MultiLabelDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # 멀티 레이블인 경우 float형으로 변경
            return item

        def __len__(self):
            return len(self.labels)

    # 데이터셋 생성
    train_dataset = MultiLabelDataset(train_encodings, Y_train)
    eval_dataset = MultiLabelDataset(eval_encodings, Y_eval)

    # Trainer 설정
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_total_limit=3,
        save_steps=1000,
        learning_rate=5e-5,
        bf16=torch.cuda.is_available(),  # FP16 활성화 GPU의 경우
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,  # 학습률
        weight_decay=0.01  # 가중치 감쇠(L2 정규화)
    )

    num_train_epochs = training_args.num_train_epochs
    num_training_steps = num_train_epochs * len(train_dataset) // training_args.per_device_train_batch_size
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # compute_metrics 함수 최적화
    def compute_metrics(p: EvalPrediction):
        preds = (torch.sigmoid(torch.tensor(p.predictions)) >= 0.5).cpu().numpy()
        labels = p.label_ids
    
        # 샘플별 F1 스코어
        f1 = f1_score(labels, preds, average='samples', zero_division=0)
        # 햄밍 손실
        hamming = hamming_loss(labels, preds)
    
        return {
            'f1_samples': f1,
            'hamming_loss': hamming,
        }
    
    # Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler)
    )
    
    # 모델 학습
    checkpoint_dir = './results'
    checkpoint_folders = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('checkpoint-')
    ]
    
    if checkpoint_folders:
        # 체크포인트를 숫자 순으로 정렬하여 가장 최근 체크포인트 선택
        checkpoint_folders.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoint_folders[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("No checkpoints found. Starting training from scratch.")
        trainer.train()

    # 모델 평가
    eval_results = trainer.evaluate()
    print(eval_results)

    # 테스트 데이터셋 평가
    predictions = trainer.predict(eval_dataset)
    pred_labels = (torch.sigmoid(torch.tensor(predictions.predictions)) >= 0.5).cpu().numpy()
    
    test_labels_np = np.vstack(eval_dataset.labels)

    # Classification report 출력
    print(classification_report(test_labels_np, pred_labels, target_names=label_categories))

    # 모델 및 토크나이저 저장
    model_save_path = "./saved_model"
    os.makedirs(model_save_path, exist_ok=True)

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model and tokenizer saved to {model_save_path}")