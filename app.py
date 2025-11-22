import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('tumor_model.h5')
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Web page layout
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to predict tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0]

    # Show result
    st.subheader(f"Predicted Tumor Type: {class_names[predicted_class]}")
    st.write("Confidence Scores:")
    st.bar_chart({class_names[i]: confidence[i] for i in range(4)})