import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_names = ['alcantarilla', 'baches', 'esquinas', 'grietas_bloque', 'lon_transv', 'parche', 'superficial']

def predict_single_image_h5(img_path):
    model = load_model('models/CNN2/CNN2_Model.h5')
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]