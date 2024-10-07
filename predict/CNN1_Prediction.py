import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Transformación de la imagen
new_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Definición del modelo
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cargar el modelo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = NeuralNet()
net.load_state_dict(torch.load('models/CNN1/CNN1_Model.pth', map_location=device))
net.to(device)
net.eval()  # Poner el modelo en modo evaluación

# Lista de nombres de clases
class_names = ['alcantarilla', 'baches', 'esquinas', 'grietas_bloque', 'lon_transv', 'parche', 'superficial']

# Función de predicción
def predict(image_path):
    image = Image.open(image_path)
    image_transformed = new_transform(image).unsqueeze(0)  # Agregar una dimensión para el batch
    image_transformed = image_transformed.to(device)

    with torch.no_grad():
        outputs = net(image_transformed)
        _, predicted = torch.max(outputs.data, 1)

    # Retornar el nombre de la clase correspondiente
    return class_names[predicted.item()]