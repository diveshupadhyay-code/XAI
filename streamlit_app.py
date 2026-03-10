import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import gdown
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # Purane model ko naye system mein chalane ka jugaad
import tensorflow as tf

# --- 1. Model Loading Ka Jugad ---
file_id = '1iKNJyPhAKIRrzR7ihhNgxrgzgSPQS-8y'
url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'xai_pneumonia_model.h5'

@st.cache_resource
def load_my_model():
    if not os.path.exists(model_path):
        with st.spinner('Bhai, model download ho raha hai... thoda sabar rakh!'):
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# --- 2. UI Layout ---
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")
st.title("🩺 Pneumonia Detection AI")
st.write("Apna X-ray upload karo, tera bhai bataiga kya scene hai!")

# --- 3. Image Upload ---
file = st.file_uploader("X-ray image (JPG/PNG) yahan dalo...", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Bhai, pehle photo toh daal!")
else:
    # Photo dikhao
    image = Image.open(file)
    st.image(image, caption='Tera X-ray', use_container_width=True)

    # --- 4. Pre-processing (Asli Kaam) ---
    def import_and_predict(image_data, model):
        size = (224, 224) # Check kar lena tere model ka input size yahi hai na?
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img_array = np.asarray(image)
        
        # Agar image grayscale hai toh 3 channels mein convert karo
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
            
        img_reshape = img_array[np.newaxis, ...]
        img_reshape = img_reshape / 255.0 # Normalization (zaruri hai!)
        
        prediction = model.predict(img_reshape)
        return prediction

    # --- 5. Result Display ---
    if st.button("Check Karo 🚀"):
        result = import_and_predict(image, model)
        
        # Maano agar model binary classification kar raha hai (0 = Normal, 1 = Pneumonia)
        if result[0][0] > 0.5:
            st.error(f"⚠️ O teri! Pneumonia ke chances lag rahe hain. (Score: {result[0][0]:.2f})")
            st.warning("Doctor se mil le jaldi, tension nahi lene ka!")
        else:
            st.success(f"✅ Ekdum fit hai boss! Normal report lag rahi hai. (Score: {result[0][0]:.2f})")
            st.balloons()
