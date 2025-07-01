import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import spacy
import networkx as nx
from tqdm import tqdm

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

def get_num_classes(train_df, test_df):
    """Get the total number of unique classes in the dataset"""
    all_relations = pd.concat([train_df['relation'], test_df['relation']]).unique()
    return len(all_relations)

class RelationExtractionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def pad_adjacency_matrix(self, adj_matrix, max_size):
        """Pad adjacency matrix to max_size"""
        current_size = adj_matrix.shape[0]
        if current_size >= max_size:
            return adj_matrix[:max_size, :max_size]
        
        # Pad with zeros
        padded = torch.zeros((max_size, max_size))
        padded[:current_size, :current_size] = adj_matrix
        return padded
    
    def create_adjacency_matrix(self, sentence):
        # Parse sentence using spaCy
        doc = nlp(sentence)
        
        # Create adjacency matrix from dependency parse
        edges = []
        for token in doc:
            edges.append((token.i, token.head.i))
        
        # Create graph and its adjacency matrix
        G = nx.Graph(edges)
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix = torch.FloatTensor(adj_matrix)
        
        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
        
        # Normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_matrix = torch.mm(torch.mm(degree_inv_sqrt, adj_matrix), degree_inv_sqrt)
        
        return self.pad_adjacency_matrix(adj_matrix, self.max_length)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # Tokenize sentence
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix(sentence)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'adj_matrix': adj_matrix,
            'label': torch.tensor(label, dtype=torch.long)
        }

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.3)  # moderate dropout rate
        self.batch_norm = nn.BatchNorm1d(out_features)
        
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        output = F.relu(output)
        # Batch norm expects (batch*nodes, features)
        b, n, f = output.shape
        output = self.batch_norm(output.view(-1, f)).view(b, n, f)
        return self.dropout(output)

class RelationExtractionGCN(nn.Module):
    def __init__(self, num_classes, bert_model='bert-base-uncased'):
        super(RelationExtractionGCN, self).__init__()
        
        # BERT layers (fine-tuning allowed)
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)  # dropout after BERT
        
        # GCN layers
        self.gcn1 = GCNLayer(self.bert_dim, 512)
        self.gcn2 = GCNLayer(512, 256)
        
        # Classification layers
        self.classifier_dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, input_ids, attention_mask, adj_matrix):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout to BERT embeddings
        sequence_output = self.dropout(sequence_output)
        
        # Apply GCN layers
        x = self.gcn1(sequence_output, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        
        # Global average pooling over tokens
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier_dropout(x)
        logits = self.classifier(x)
        return logits

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping_patience=3):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, adj_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                adj_matrix = batch['adj_matrix'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, adj_matrix)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model_final.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered")
                break


def main():
    # Load and preprocess data
    # train_df = pd.read_parquet('train-00000-of-00001.parquet')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #For Data augmentation
    train_dataset = load_from_disk("sr_augmented_data/augmented_dataset.arrow")
    test_df = pd.read_parquet('test-00000-of-00001.parquet')
    
    # Convert Hugging Face dataset to Pandas DataFrame
    train_df = train_dataset.to_pandas()

    # Get number of classes
    print(train_df)
    num_classes = get_num_classes(train_df, test_df)
    print(f"Number of unique relations in dataset: {num_classes}")
    
    print("\nUnique relations:")
    print(pd.concat([train_df['relation'], test_df['relation']]).unique())
    
    # Split training data into train and validation sets
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize model and training components
    
    model = RelationExtractionGCN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use two parameter groups: one for BERT (lower LR) and one for the rest (higher LR)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': [p for name, p in model.named_parameters() if 'bert' not in name], 'lr': 2e-5}
    ], weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # Train model with early stopping
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=5, device=device, early_stopping_patience=3)
    
    
if __name__ == "__main__":
    main()