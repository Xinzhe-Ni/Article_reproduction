#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = " "   # 指定GPU
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset


# In[ ]:


### 读取数据集（注意这里需要打开VPN）

train_dataset = load_dataset("ag_news", split="train[:4000]")
val_dataset = load_dataset("ag_news", split="train[4000:5000]")
test_dataset = load_dataset("ag_news", split="test")

train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)   # BERT期望的标签字段是labels而不是label，因此简单处理一下
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)


# In[ ]:


### 加载模型，tokenizer，并对数据进行预处理

model_id = 'prajjwal1/bert-tiny'
model = AutoModelForSequenceClassification.from_pretrained(model_id, 
            num_labels=train_dataset.features["label"].num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_id)

MAX_LENGTH = 256   #把所有序列统一到256个token
train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])   # 为了在pytorch训练，还需要声明格式和字段
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


# In[ ]:


### 模型训练

def compute_metrics(pred):   # 定义指标
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',          
    learning_rate=3e-4,
    num_train_epochs=10,              
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,   
    logging_dir='./logs',            
    logging_steps=100,
    save_strategy="epoch",
    do_train=True,
    do_eval=True,
    no_cuda=True,
    load_best_model_at_end=True,
    # eval_steps=100,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,               
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,       
    compute_metrics=compute_metrics
)

train_out = trainer.train()


# In[ ]:


### 模型预测

model = model.cpu()
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
test_examples = load_dataset("ag_news", split="test[:10]")
test_examples.shape

result = classifier(test_examples[0]['text'])

