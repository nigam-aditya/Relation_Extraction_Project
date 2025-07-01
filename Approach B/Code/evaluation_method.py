import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import spacy
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

class RelationExtractionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def pad_adjacency_matrix(self, adj_matrix, max_size):
        current_size = adj_matrix.shape[0]
        if current_size >= max_size:
            return adj_matrix[:max_size, :max_size]
        padded = torch.zeros((max_size, max_size))
        padded[:current_size, :current_size] = adj_matrix
        return padded
    
    def create_adjacency_matrix(self, sentence):
        doc = nlp(sentence)
        edges = []
        for token in doc:
            edges.append((token.i, token.head.i))
        
        G = nx.Graph(edges)
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix = torch.FloatTensor(adj_matrix)
        
        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
        degree = torch.sum(adj_matrix, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_matrix = torch.mm(torch.mm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
        
        return self.pad_adjacency_matrix(adj_matrix, self.max_length)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        adj_matrix = self.create_adjacency_matrix(sentence)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'adj_matrix': adj_matrix,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Revised GCNLayer including batch normalization
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(out_features)
        
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        output = F.relu(output)
        b, n, f = output.shape
        # BatchNorm expects a 2D tensor: (batch_size * nodes, features)
        output = self.batch_norm(output.view(-1, f)).view(b, n, f)
        return self.dropout(output)

class RelationExtractionGCN(nn.Module):
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(RelationExtractionGCN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size
        self.gcn1 = GCNLayer(self.bert_dim, 512)
        self.gcn2 = GCNLayer(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask, adj_matrix):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        x = self.dropout(sequence_output)
        x = self.gcn1(x, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        x = torch.mean(x, dim=1)  # Global average pooling over tokens
        logits = self.classifier(x)
        return logits

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask, adj_matrix)
            _, predictions = outputs.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_predictions), np.array(all_labels)

def analyze_errors(sentences, true_labels, predictions, relation_names):
    errors = []
    for sent, true, pred in zip(sentences, true_labels, predictions):
        if true != pred:
            errors.append({
                'sentence': sent,
                'true_relation': relation_names.get(true, f"Relation-{true}"),
                'predicted_relation': relation_names.get(pred, f"Relation-{pred}")
            })
    return pd.DataFrame(errors)

def main():
    # Load training and test data
    print("Loading train data...")
    train_df = pd.read_parquet('train-00000-of-00001.parquet')
    print(f"Train data shape: {train_df.shape}")
    print("Loading test data...")
    test_df = pd.read_parquet('test-00000-of-00001.parquet')
    print(f"Test data shape: {test_df.shape}")
    
    # Split train data into training and validation sets
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Determine unique relations across train and test
    unique_relations = sorted(np.unique(np.concatenate([train_df['relation'].unique(), test_df['relation'].unique()])))
    num_classes = len(unique_relations)
    print(f"\nFound {num_classes} unique relations: {unique_relations}")
    
    # Create relation mapping for error analysis and reporting
    relation_names = {i: f"Relation-{i}" for i in unique_relations}
    
    print("\nInitializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("\nCreating datasets...")
    train_dataset = RelationExtractionDataset(
        train_data['sentence'].tolist(),
        train_data['relation'].tolist(),
        tokenizer
    )
    val_dataset = RelationExtractionDataset(
        val_data['sentence'].tolist(),
        val_data['relation'].tolist(),
        tokenizer
    )
    test_dataset = RelationExtractionDataset(
        test_df['sentence'].tolist(),
        test_df['relation'].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nLoading model...")
    model = RelationExtractionGCN(num_classes=num_classes).to(device)
    try:
        # Set weights_only=True to suppress pickle warnings.
        checkpoint = torch.load('best_model_final.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
        if 'epoch' in checkpoint and 'val_loss' in checkpoint:
            print(f"Checkpoint from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate on training set
    print("\nEvaluating training set...")
    train_predictions, train_true_labels = evaluate_model(model, train_loader, device)
    train_report = classification_report(train_true_labels, train_predictions, 
                                           labels=unique_relations,
                                           target_names=[relation_names[i] for i in unique_relations],
                                           digits=4, zero_division=0)
    print("\nTraining Set Classification Report:")
    print(train_report)
    
    # Generate confusion matrix for training set
    print("\nGenerating confusion matrix for training set...")
    cm_train = confusion_matrix(train_true_labels, train_predictions, labels=unique_relations)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=[relation_names[i] for i in unique_relations],
                yticklabels=[relation_names[i] for i in unique_relations])
    plt.title('Training Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('train_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze errors for training set
    print("\nAnalyzing errors for training set...")
    error_df_train = analyze_errors(train_data['sentence'].tolist(), train_true_labels, train_predictions, relation_names)
    error_df_train.to_csv('train_error_analysis.csv', index=False)
    print(f"\nFound {len(error_df_train)} errors out of {len(train_data)} train samples")
    print(f"Overall train error rate: {(len(error_df_train)/len(train_data)*100):.2f}%")
    
    # Evaluate on validation set
    print("\nEvaluating validation set...")
    val_predictions, val_true_labels = evaluate_model(model, val_loader, device)
    val_report = classification_report(val_true_labels, val_predictions, 
                                         labels=unique_relations,
                                         target_names=[relation_names[i] for i in unique_relations],
                                         digits=4, zero_division=0)
    print("\nValidation Set Classification Report:")
    print(val_report)
    
    # Generate confusion matrix for validation set
    print("\nGenerating confusion matrix for validation set...")
    cm_val = confusion_matrix(val_true_labels, val_predictions, labels=unique_relations)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=[relation_names[i] for i in unique_relations],
                yticklabels=[relation_names[i] for i in unique_relations])
    plt.title('Validation Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('val_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze errors for validation set
    print("\nAnalyzing errors for validation set...")
    error_df_val = analyze_errors(val_data['sentence'].tolist(), val_true_labels, val_predictions, relation_names)
    error_df_val.to_csv('val_error_analysis.csv', index=False)
    print(f"\nFound {len(error_df_val)} errors out of {len(val_data)} validation samples")
    print(f"Overall validation error rate: {(len(error_df_val)/len(val_data)*100):.2f}%")
    
    # Evaluate on test set
    print("\nEvaluating test set...")
    test_predictions, test_true_labels = evaluate_model(model, test_loader, device)
    test_report = classification_report(test_true_labels, test_predictions, 
                                          labels=unique_relations,
                                          target_names=[relation_names[i] for i in unique_relations],
                                          digits=4, zero_division=0)
    print("\nTest Set Classification Report:")
    print(test_report)
    
    # Generate confusion matrix for test set
    print("\nGenerating confusion matrix for test set...")
    cm_test = confusion_matrix(test_true_labels, test_predictions, labels=unique_relations)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=[relation_names[i] for i in unique_relations],
                yticklabels=[relation_names[i] for i in unique_relations])
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze errors for test set
    print("\nAnalyzing errors for test set...")
    error_df_test = analyze_errors(test_df['sentence'].tolist(), test_true_labels, test_predictions, relation_names)
    error_df_test.to_csv('test_error_analysis.csv', index=False)
    print(f"\nFound {len(error_df_test)} errors out of {len(test_df)} test samples")
    print(f"Overall test error rate: {(len(error_df_test)/len(test_df)*100):.2f}%")
    
    print("\nEvaluation complete. Analysis files saved:")
    print("1. Classification reports (printed to console)")
    print("2. train_confusion_matrix.png, val_confusion_matrix.png, test_confusion_matrix.png")
    print("3. train_error_analysis.csv, val_error_analysis.csv, test_error_analysis.csv")

if __name__ == "__main__":
    main()
