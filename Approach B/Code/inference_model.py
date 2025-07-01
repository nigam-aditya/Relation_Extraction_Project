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

def load_model(model_path, num_classes, device):
    """
    Load the trained GCN+BERT model from a saved .pt file.
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        num_classes (int): Number of relation classes.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model (RelationExtractionGCN): Loaded model ready for inference.
        tokenizer (BertTokenizer): Tokenizer for text preprocessing.
    """
    # Initialize model
    model = RelationExtractionGCN(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print(f"Model successfully loaded from {model_path} (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    
    return model, tokenizer


def predict_relation_from_text(model, tokenizer, sentence, device, max_length=128):
    """
    Predict the relation for a given sentence and return the predicted label along with its confidence score.
    
    Args:
        model (RelationExtractionGCN): Trained relation extraction model.
        tokenizer (BertTokenizer): Tokenizer for processing input text.
        sentence (str): Input sentence containing two tagged entities (<e1> and <e2>).
        device (str): Device to run the inference on.
        max_length (int): Maximum token length for sentence padding/truncation.
        
    Returns:
        predicted_relation (int): The predicted relation class.
        confidence (float): Confidence score of the prediction.
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Prepare input using the RelationExtractionDataset class
    dataset = RelationExtractionDataset([sentence], [0], tokenizer, max_length=max_length)
    batch = dataset[0]
    
    input_ids = batch['input_ids'].unsqueeze(0).to(device)
    attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
    adj_matrix = batch['adj_matrix'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, adj_matrix)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        
    return predicted.item(), confidence.item()


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset to get num_classes dynamically
    test_df = pd.read_parquet('test-00000-of-00001.parquet')
    num_classes = len(test_df['relation'].unique())

    # Load model
    model, tokenizer = load_model('best_model_final.pt', num_classes, device)

    # Example inference
    test_sentence = "The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
    predicted_relation, confidence = predict_relation_from_text(model, tokenizer, test_sentence, device)

    print(f"\nPredicted relation: {predicted_relation} with confidence: {confidence:.4f}")
