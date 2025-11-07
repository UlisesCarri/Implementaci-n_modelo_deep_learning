import torch
import torch.nn as nn
import pandas   as pd
from torch.utils.data           import DataLoader
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from torch.nn.utils.rnn         import pad_sequence
from collections                import Counter
from Model_Components           import *
import pickle

#Usar gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Hiperparámetros
VOCAB_SIZE    = 70000
Dim_Embedding = 100
HIDDEN_DIM    = 128
Dropout  = 0.5
Batch_tam    = 32
Epochs    = 10
DATA          = r"emotions.csv"

def Gen_Model(DATA,Dim_Embedding,HIDDEN_DIM,Dropout,Batch_tam,Epochs):
    # Recuperar csv
    df = pd.read_csv(DATA)

    # Preprocesamiento de etiquetas
    encoder     = LabelEncoder()
    df['label'] = encoder.fit_transform(df["label"])
    N_Clases = len(encoder.classes_)

    train_val_df, test_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['label']
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']
    )

    # Contar todas las palabras
    all_tokens = [token for text in train_df["text"] for token in tokenize(text)]
    word_counts = Counter(all_tokens)

    # Crear el vocabulario (mapa de palabra a índice)
    vocab = {"<pad>": 0, "<unk>": 1} #Agregar por defecto pading y desconocido
    idx_counter = 2

    # Solo añadir al vocabulario las palabras mas comunes
    for word, _ in word_counts.most_common(VOCAB_SIZE - 2):
        vocab[word] = idx_counter
        idx_counter += 1

    print(f"Tamaño del vocabulario: {len(vocab)}")

    # Función Collate para Padding Dinámico (Crucial en PyTorch)
    def collate_fn(batch):
        # batch es una lista de (sequence, label)
        sequences = [item[0] for item in batch]
        labels    = [item[1] for item in batch]

        # Aplicar padding: sequences ahora tienen el mismo largo (el largo del más largo del batch)
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=vocab["<pad>"])
        return sequences_padded.to(device), torch.stack(labels).to(device)

    # Crear instancias del Dataset
    train_dataset = EmotionDataset(train_df, vocab)
    val_dataset   = EmotionDataset(val_df  , vocab)
    test_dataset  = EmotionDataset(test_df , vocab)

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Batch_tam, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset  , batch_size=Batch_tam, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset , batch_size=Batch_tam, shuffle=False, collate_fn=collate_fn)

    # Inicializar el modelo
    model = GRUClassifier(len(vocab), Dim_Embedding, HIDDEN_DIM, 1, N_Clases, Dropout).to(device)

    # Definir la función de pérdida y el optimizador
    criterion = nn.CrossEntropyLoss() #problema de clasificacion multimple entonces se usa Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #usamos adam como optimizador cun un step de 0.001
    
    train_loss = []
    val_loss   = []
    min        = 100
    print("\nIniciando el entrenamiento...")
    for epoch in range(Epochs):
        trainL     = train(model, train_loader, criterion, optimizer)
        valL, valA = evaluate(model, val_loader, criterion)

        if valL < min:
            min    = valL
            final_Mod = model
        train_loss.append(trainL)
        val_loss.append(valL)
        
        print(f"Época {epoch+1}/{Epochs} | Pérdida Train: {trainL:.4f} | Pérdida Val: {valL:.4f} | Precisión Val: {valA:.4f}")

    valL, valA = evaluate(final_Mod, val_loader, criterion)
    print(f"Pérdida Val: {valL:.4f} | "f"Precisión Val: {valA:.4f}")
    testL, testA = evaluate(final_Mod, test_loader, criterion)
    print("\n\n=========TEST=========")
    print(f"Pérdida Test: {testL:.4f} | "f"Precisión Test: {testA:.4f}")

    plot(train_loss,val_loss)
    try:
        torch.save(final_Mod.state_dict(), "GruNet.pt") #Salvar pesos de modelo
        with open("vocabV2.pkl", "wb") as f: #Salvar el vocabulario
            pickle.dump(vocab, f)
        with open("label_encoderV2.pkl", "wb") as f: #Salvar encoder
            pickle.dump(encoder, f)

        print("El modelo se guardo con exito :D")
    except:
        print("El modelo fallo en el proceso de guardado :(")

Gen_Model(DATA,Dim_Embedding,HIDDEN_DIM,Dropout,Batch_tam,Epochs)