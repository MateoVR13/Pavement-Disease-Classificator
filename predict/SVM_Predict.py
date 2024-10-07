import joblib
import numpy as np
import cv2

# Cargar el modelo SVM
svm_loaded = joblib.load('models/SVM_Model.pkl')

# Cargar PCA
pca_loaded = joblib.load('models/SVM_PCA.pkl')

# Cargar LabelBinarizer
lb = joblib.load('models/SVM_Label_Binarizer.pkl')  # Asegúrate de que el archivo existe

# Función para preprocesar la imagen
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Redimensionar la imagen
    image = image / 255.0  # Normalizar
    return image.reshape(1, -1)  # Convertir a formato plano

def predict(image_path):
    # Cargar y preprocesar la nueva imagen
    new_image = preprocess_image(image_path)

    # Reducir dimensionalidad con PCA
    new_image_pca = pca_loaded.transform(new_image)

    # Realizar la predicción
    predicted_class = svm_loaded.predict(new_image_pca)
    predicted_label = lb.classes_[predicted_class[0]]  # Asegúrate de que lb esté definido

    return predicted_label
