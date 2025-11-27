"""
Complete Training Script for Fake News Detection with GPU Acceleration
Models: Logistic Regression, Naive Bayes, BiLSTM, BERT
Datasets: PolitiFact + GossipCop
GPU-Optimized with Mixed Precision Training
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from collections import Counter
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# GPU Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = torch.cuda.is_available()  # Automatic mixed precision for faster training

print("="*80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Mixed Precision Training: {'ENABLED' if USE_MIXED_PRECISION else 'DISABLED'}")
print("="*80 + "\n")

# Dataset paths
POLITIFACT_REAL = r"D:\fakeNewsPoli\politifact_real.csv"
POLITIFACT_FAKE = r"D:\fakeNewsPoli\politifact_fake.csv"
GOSSIPCOP_REAL = r"D:\fakeNewsPoli\gossipcop_real.csv"
GOSSIPCOP_FAKE = r"D:\fakeNewsPoli\gossipcop_fake.csv"

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 32 if torch.cuda.is_available() else 16  # Larger batch on GPU
BERT_EPOCHS = 5
BILSTM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500

# BiLSTM hyperparameters
VOCAB_SIZE = 20000
EMBEDDING_DIM = 200
HIDDEN_DIM = 128
BILSTM_LAYERS = 2
DROPOUT = 0.5

# ============================================================================
# STEP 1: LOAD AND PREPARE DATASETS
# ============================================================================
def load_and_prepare_data():
    """Load all datasets and combine them"""
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    # Load datasets
    politifact_real = pd.read_csv(POLITIFACT_REAL)
    politifact_fake = pd.read_csv(POLITIFACT_FAKE)
    gossipcop_real = pd.read_csv(GOSSIPCOP_REAL)
    gossipcop_fake = pd.read_csv(GOSSIPCOP_FAKE)
    
    # Add labels (0 = real, 1 = fake)
    politifact_real['label'] = 0
    politifact_fake['label'] = 1
    gossipcop_real['label'] = 0
    gossipcop_fake['label'] = 1
    
    # Combine datasets
    all_data = pd.concat([
        politifact_real, politifact_fake,
        gossipcop_real, gossipcop_fake
    ], ignore_index=True)
    
    # Use 'title' column for text
    all_data = all_data[['title', 'label']].rename(columns={'title': 'text'})
    all_data = all_data.dropna(subset=['text'])
    
    # Balance dataset
    fake_data = all_data[all_data['label'] == 1]
    real_data = all_data[all_data['label'] == 0]
    min_count = min(len(fake_data), len(real_data))
    
    fake_balanced = fake_data.sample(n=min_count, random_state=42)
    real_balanced = real_data.sample(n=min_count, random_state=42)
    balanced_data = pd.concat([fake_balanced, real_balanced], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Fake news: {(balanced_data['label'] == 1).sum()}")
    print(f"Real news: {(balanced_data['label'] == 0).sum()}")
    
    return balanced_data

# ============================================================================
# STEP 2: TRAIN BASELINE MODELS (Logistic Regression & Naive Bayes)
# ============================================================================
def train_baseline_models(data):
    """Train Logistic Regression and Naive Bayes models"""
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS (Logistic Regression & Naive Bayes)")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], 
        test_size=0.2, random_state=42, stratify=data['label']
    )
    
    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # Save vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ TF-IDF vectorizer saved")
    
    results = {}
    
    # ========================================================================
    # Logistic Regression
    # ========================================================================
    print("\n--- Training Logistic Regression ---")
    start_time = time.time()
    
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        solver='lbfgs'
    )
    lr_model.fit(X_train_tfidf, y_train)
    
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(
        y_test, lr_pred, average='binary'
    )
    
    training_time = time.time() - start_time
    
    print(f"\nLogistic Regression Results:")
    print(f"  Accuracy:  {lr_acc:.4f}")
    print(f"  Precision: {lr_prec:.4f}")
    print(f"  Recall:    {lr_rec:.4f}")
    print(f"  F1-Score:  {lr_f1:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    # Save model
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("✓ Model saved as 'logistic_regression_model.pkl'")
    
    results['Logistic Regression'] = {
        'accuracy': lr_acc, 'precision': lr_prec,
        'recall': lr_rec, 'f1': lr_f1
    }
    
    # ========================================================================
    # Naive Bayes
    # ========================================================================
    print("\n--- Training Naive Bayes ---")
    start_time = time.time()
    
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)
    
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_prec, nb_rec, nb_f1, _ = precision_recall_fscore_support(
        y_test, nb_pred, average='binary'
    )
    
    training_time = time.time() - start_time
    
    print(f"\nNaive Bayes Results:")
    print(f"  Accuracy:  {nb_acc:.4f}")
    print(f"  Precision: {nb_prec:.4f}")
    print(f"  Recall:    {nb_rec:.4f}")
    print(f"  F1-Score:  {nb_f1:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    # Save model
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print("✓ Model saved as 'naive_bayes_model.pkl'")
    
    results['Naive Bayes'] = {
        'accuracy': nb_acc, 'precision': nb_prec,
        'recall': nb_rec, 'f1': nb_f1
    }
    
    return results

# ============================================================================
# STEP 3: BiLSTM MODEL
# ============================================================================
class BiLSTMDataset(Dataset):
    """Dataset for BiLSTM model"""
    def __init__(self, texts, labels, vocab_dict, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab_dict = vocab_dict
        self.max_length = max_length
        self.unk_idx = vocab_dict.get('<UNK>', 1)
        self.pad_idx = vocab_dict.get('<PAD>', 0)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower().split()
        label = self.labels[idx]
        
        # Convert words to indices
        indices = [self.vocab_dict.get(word, self.unk_idx) for word in text]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.pad_idx] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BiLSTMModel(nn.Module):
    """BiLSTM model for text classification"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=2, 
                 n_layers=2, dropout=0.5, pad_idx=0):
        super(BiLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        x = self.fc1(hidden)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

def create_vocab(texts, vocab_size=20000):
    """Create vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(str(text).lower().split())
    
    most_common = word_counts.most_common(vocab_size - 2)
    vocab_dict = {'<PAD>': 0, '<UNK>': 1}
    
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab_dict[word] = idx
    
    return vocab_dict

def train_bilstm(data):
    """Train BiLSTM model with GPU acceleration"""
    print("\n" + "="*80)
    print("TRAINING BiLSTM MODEL (GPU Accelerated)")
    print("="*80)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(), data['label'].tolist(),
        test_size=0.2, random_state=42, stratify=data['label']
    )
    
    # Create vocabulary
    print("\nCreating vocabulary...")
    vocab_dict = create_vocab(train_texts, VOCAB_SIZE)
    print(f"✓ Vocabulary size: {len(vocab_dict)}")
    
    # Save vocabulary
    with open('bilstm_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)
    print("✓ Vocabulary saved")
    
    # Create datasets
    train_dataset = BiLSTMDataset(train_texts, train_labels, vocab_dict, MAX_LENGTH)
    val_dataset = BiLSTMDataset(val_texts, val_labels, vocab_dict, MAX_LENGTH)
    
    # Use pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           pin_memory=True, num_workers=0)
    
    # Initialize model
    model = BiLSTMModel(
        vocab_size=len(vocab_dict),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=BILSTM_LAYERS,
        dropout=DROPOUT
    )
    model.to(DEVICE)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"\nTraining for {BILSTM_EPOCHS} epochs on {DEVICE}...")
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(BILSTM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                
                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{BILSTM_EPOCHS} - "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'bilstm_model_cross_domain.pt')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
    
    training_time = time.time() - start_time
    print(f"\n✓ BiLSTM training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"✓ Training time: {training_time/60:.2f} minutes")
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    final_acc = accuracy_score(all_labels, all_preds)
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    results = {
        'accuracy': final_acc, 'precision': final_prec,
        'recall': final_rec, 'f1': final_f1
    }
    
    return results

# ============================================================================
# STEP 4: BERT MODEL WITH MIXED PRECISION
# ============================================================================
class BERTDataset(Dataset):
    """Dataset for BERT model"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CustomBERTModel(nn.Module):
    """Custom BERT model with additional layers"""
    def __init__(self, num_labels=2, dropout=0.3):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        x = self.fc1(pooled_output)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

def train_bert(data):
    """Train BERT model with GPU acceleration and mixed precision"""
    print("\n" + "="*80)
    print("TRAINING BERT MODEL (GPU + Mixed Precision)")
    print("="*80)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(), data['label'].tolist(),
        test_size=0.2, random_state=42, stratify=data['label']
    )
    
    print(f"\nTrain samples: {len(train_texts)}, Val samples: {len(val_texts)}")
    
    # Initialize tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = BERTDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = BERTDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Use pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           pin_memory=True, num_workers=0)
    
    # Initialize model
    print("Initializing BERT model...")
    model = CustomBERTModel(num_labels=2)
    model.to(DEVICE)
    
    # Use PyTorch's AdamW instead of transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    # Training loop
    print(f"\nTraining for {BERT_EPOCHS} epochs on {DEVICE}...")
    if USE_MIXED_PRECISION:
        print("✓ Mixed Precision Training ENABLED (FP16)")
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(BERT_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids, attention_mask)
                
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{BERT_EPOCHS}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'bert_model_cross_domain.pt')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
    
    training_time = time.time() - start_time
    print(f"\n✓ BERT training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"✓ Training time: {training_time/60:.2f} minutes")
    
    # Final evaluation
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    results = {
        'accuracy': best_val_acc, 'precision': final_prec,
        'recall': final_rec, 'f1': final_f1
    }
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main training pipeline"""
    overall_start = time.time()
    
    print("="*80)
    print("FAKE NEWS DETECTION - GPU ACCELERATED TRAINING PIPELINE")
    print("="*80)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load data
    data = load_and_prepare_data()
    
    # Train all models
    all_results = {}
    
    # 1. Baseline models
    baseline_results = train_baseline_models(data)
    all_results.update(baseline_results)
    
    # 2. BiLSTM
    bilstm_results = train_bilstm(data)
    all_results['BiLSTM'] = bilstm_results
    
    # 3. BERT
    bert_results = train_bert(data)
    all_results['BERT'] = bert_results
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df[['accuracy', 'precision', 'recall', 'f1']]
    print("\n" + results_df.to_string())
    
    # Save results
    results_df.to_csv('training_results_comparison.csv')
    print("\n✓ Results saved to 'training_results_comparison.csv'")
    
    total_time = time.time() - overall_start
    print(f"\n✓ Total pipeline time: {total_time/60:.2f} minutes")
    
    print("\n" + "="*80)
    print("SAVED MODELS:")
    print("="*80)
    print("✓ logistic_regression_model.pkl")
    print("✓ naive_bayes_model.pkl")
    print("✓ bilstm_model_cross_domain.pt")
    print("✓ bert_model_cross_domain.pt")
    print("✓ tfidf_vectorizer.pkl")
    print("✓ bilstm_vocab.pkl")
    
    # GPU memory summary
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU MEMORY USAGE")
        print("="*80)
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
