import torch
import torch.nn as nn
import re
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def tokenize(text): #Limpiar y separar en tokens
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def train(model, loader, criterion, optimizer):
    model.train() #colocar el modelo en modo de entrenamiento
    total_loss = 0
    for texts, labels in loader:
        optimizer.zero_grad() #reinciar el gradiente
        predictions = model(texts) #hacer prediccion con texto
        loss        = criterion(predictions, labels) #hacer CE con prediccion contra valor real
        loss.backward() #Calcular gradientes
        optimizer.step() #Optimizar pesos
        total_loss += loss.item() #sumar la perdida al total
    return total_loss / len(loader) #promediar perdida total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for texts, labels in loader:
            predictions = model(texts) #hacer prediccion con texto
            loss        = criterion(predictions, labels) #hacer CE con prediccion contra valor real
            total_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            total       += labels.size(0)
            correct     += (predicted == labels).sum().item()
            
    accuracy = correct / total #Calcular la precicion
    return total_loss / len(loader), accuracy

def sequencer(text,vocab): # Generar Secuencias de los textos
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def plot(train,val): #Graficar perdida de train y val
    epochs = range(1, len(train) + 1)

    plt.plot(epochs, train, 'o-', label='Train loss')
    plt.plot(epochs, val, 's--', label='Val loss')

    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

class EmotionDataset(Dataset): 
    def __init__(self, df, vocab):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab  = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_seq = sequencer(self.texts[idx],self.vocab) #Generar secuencia
        label    = self.labels[idx] #Colocar su etiqueta
        return torch.tensor(text_seq, dtype=torch.long), torch.tensor(label, dtype=torch.long) #generar tensor para meter al modelo

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #capa de embedings
        self.gru = nn.GRU( #capa de RNN
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_dim, num_classes) #capa densa para calsificar

    def forward(self, text):
        embedded       = self.embedding(text)
        output, hidden = self.gru(embedded)
        hidden_final   = hidden[-1] 
        output         = self.fc(hidden_final) 
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_final = hidden[-1]
        output = self.fc(hidden_final)
        return output