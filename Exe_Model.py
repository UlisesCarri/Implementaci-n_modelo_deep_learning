import pickle
import torch
from Model_Components import GRUClassifier,sequencer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

EMBEDDING_DIM = 100
HIDDEN_DIM    = 128
NUM_LAYERS    = 1
DROPOUT_RATE  = 0
emotions      = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}

def load_model(dir_vocab,dir_enco,PT):
    global model,vocab,encoder
    with open(dir_vocab, "rb") as f: #Recuperar el vocabulario
        vocab = pickle.load(f)
    with open(dir_enco, "rb") as f: #Recuperar los encoders
        encoder = pickle.load(f)

    NUM_CLASSES   = len(encoder.classes_)
    model = GRUClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(device) #Recrerar la arquitectura del modelo
    model.load_state_dict(torch.load(PT, map_location=device)) #Cargar los pesos entrenados del .pt

def predict_emotion(text_a_predecir):
    sequence          = sequencer(text_a_predecir,vocab) #Generar la secuencia
    input_tensor      = torch.tensor([sequence], dtype=torch.long).to(device) #Crear tensor con la secuencia
    probabilities     = torch.softmax(model(input_tensor), dim=1) #Softamax para normalizar y sacar la probabilidad por clase
    predicted_index   = torch.argmax(probabilities, dim=1).item() #Sacar numero de clase mas probable

    print(f"Emoción Predicha: {emotions[int(predicted_index)]}")

load_model("vocabV3.pkl","label_encoderV3.pkl","GruNetV3.pt")
print("Emotions: sadness,joy,love,anger,fear, surprise")

while True:
    ine = input("Write a sentence with emotion(MAX 32 words):")
    predict_emotion(ine)