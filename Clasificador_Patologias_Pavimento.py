import streamlit as st
from predict.CNN1_Prediction import predict as cnn1_predict
from predict.CNN2_Prediction import predict_single_image_h5 as cnn2_predict
from predict.KNN_Predict import predict as knn_predict
from predict.SVM_Predict import predict as svm_predict
import pandas as pd

st.title("Clasificación de Patologías del Pavimento")

# Crear dos columnas con proporciones ajustadas
col1, col2 = st.columns([2, 2])

# Columna 1: Carga de imagen
with col1:
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png"])

    if uploaded_file is not None:
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(image_path, caption='Imagen subida.', use_column_width=True)

        # Botón para clasificar
        if st.button("Clasificar"):
            # Hacer las clasificaciones
            try:
                cnn1_result = cnn1_predict(image_path)
                cnn2_result = cnn2_predict(image_path)
                knn_result = knn_predict(image_path)
                svm_result = svm_predict(image_path)

                # Crear un DataFrame para los resultados
                results = {
                    "Modelo": ["CNN1", "CNN2", "KNN", "SVM"],
                    "Predicción": [cnn1_result, cnn2_result, knn_result, svm_result]
                }
                results_df = pd.DataFrame(results)

                # Mostrar resultados en la columna 2
                with col2:
                    st.subheader("Resultados de la Clasificación")
                    st.table(results_df)
            except Exception as e:
                st.error(f"Ocurrió un error durante la clasificación: {e}")

# Columna 2: (Dejamos este espacio para que sea utilizado si es necesario)
with col2:
    if uploaded_file is None:
        st.write("Sube una imagen para clasificarla.")