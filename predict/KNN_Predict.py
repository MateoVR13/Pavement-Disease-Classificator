import cv2
import joblib

class_names = ['alcantarilla', 'baches', 'esquinas', 'grietas_bloque', 'lon_transv', 'parche', 'superficial']

def load_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")

    image = cv2.resize(image, (224, 224)) / 255.0
    return image.reshape(1, -1)

def predict(image_path):
    model = joblib.load('models/KNN_Model.pkl')
    pca = joblib.load('models/KNN_PCA_Model.pkl')
    lb = joblib.load('models/KNN_Label_Binarizer.pkl')

    X_single = load_single_image(image_path)
    X_single_pca = pca.transform(X_single)
    y_pred_single = model.predict(X_single_pca)

    return lb.classes_[y_pred_single[0]]