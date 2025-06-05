import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os

# 📥 Download model dari Google Drive kalau belum ada
model_path = "waste_classifier_model.h5"
if not os.path.exists(model_path):
    gdown.download("https://drive.google.com/file/d/14vWe25RkQcoAbWz9IZwYr2DnPCt6QzNW/view?usp=sharing", model_path, quiet=False)
    
# Load model
model = load_model('waste_classifier_model.h5')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Ganti sesuai class kamu

img_height, img_width = 224, 224

st.title("Klasifikasi Sampah Otomatis")
st.write("Upload gambar sampah, dan model akan memprediksi jenisnya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(img_height, img_width))
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"### Prediksi: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")
