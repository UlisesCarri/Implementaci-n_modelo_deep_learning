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
DATA          = r"emotions_newV2.csv"

def Gen_Model(DATA,Dim_Embedding,HIDDEN_DIM,Dropout,Batch_tam,Epochs):
    # Recuperar csv
    df = pd.read_csv(DATA)

    # Preprocesamiento de etiquetas
    encoder     = LabelEncoder()
    df['label'] = encoder.fit_transform(df["label"])
    N_Clases = len(encoder.classes_)

    # Separar data sets
    train_val_df, test_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']
    )

    # Contar palabras
    all_tokens = [token for text in train_df["text"] for token in tokenize(text)]
    word_counts = Counter(all_tokens)

    # Crear el vocabulario
    vocab = {"<pad>": 0, "<unk>": 1} #Agregar por defecto pading y desconocido
    idx_counter = 2

    # Solo añadir al vocabulario las palabras mas comunes (al final tome el vocabulario completo pero deje eso)
    for word, _ in word_counts.most_common(VOCAB_SIZE - 2):
        vocab[word] = idx_counter
        idx_counter += 1

    print(f"Tamaño del vocabulario: {len(vocab)}")

    # Agregar padding dinámico
    def collate_fn(batch):
        # Batch es una lista de (sequence, label)
        sequences = [item[0] for item in batch] 
        labels    = [item[1] for item in batch]

        # Aplicar padding para nromalizar las entradas a el modelo
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

    # Generar modelo
    model = GRUClassifier(len(vocab), Dim_Embedding, HIDDEN_DIM, 1, N_Clases, Dropout).to(device)

    # Funcion de perdida y optimizador
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    
    train_loss = []
    val_loss   = []
    min        = 100
    print("\nIniciando el entrenamiento...")
    for epoch in range(Epochs):
        trainL     = train(model, train_loader, criterion, optimizer) #Entrenamiento
        valL, valA = evaluate(model, val_loader, criterion) #Validacion

        if valL < min: #tomar el mejor modelo
            min    = valL
            final_Mod = model

        train_loss.append(trainL) #guardar perdida
        val_loss.append(valL)
        
        print(f"Época {epoch+1}/{Epochs} | Pérdida Train: {trainL:.4f} | Pérdida Val: {valL:.4f} | Precisión Val: {valA:.4f}")

    testL, testA = evaluate(final_Mod, test_loader, criterion) #test
    print("\n\n=========TEST=========")
    print(f"Pérdida Test: {testL:.4f} | "f"Precisión Test: {testA:.4f}")

    plot(train_loss,val_loss)
    try:
        torch.save(final_Mod.state_dict(), "GruNetV3.pt") #Salvar pesos de modelo
        with open("vocabV3.pkl", "wb") as f: #Salvar el vocabulario
            pickle.dump(vocab, f)
        with open("label_encoderV3.pkl", "wb") as f: #Salvar encoder
            pickle.dump(encoder, f)

        print("El modelo se guardo con exito :D")
    except:
        print("El modelo fallo en el proceso de guardado :(")

Gen_Model(DATA,Dim_Embedding,HIDDEN_DIM,Dropout,Batch_tam,Epochs)