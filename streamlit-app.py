
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Load the class names
@st.cache_data()
def load_class_names(path):
    with open(path, 'r') as f:
        return json.load(f)

# Load the trained model
@st.cache_resource()
def load_model(path):
    return tf.keras.models.load_model(path)

# Define the paths for the model and class names
MODEL_PATH = 'cifar10_cnn_augmented_model.keras'
CLASS_NAMES_PATH = 'class_names.json'

# Ensure model and class names files exist
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure it's in the same directory.")
    st.stop()
if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"Class names file not found at {CLASS_NAMES_PATH}. Please ensure it's in the same directory.")
    st.stop()

class_names = load_class_names(CLASS_NAMES_PATH)
model = load_model(MODEL_PATH)

# Inference Pipeline Functions (copied from notebook)
def resize_image(image, target_size=(32, 32)):
    if image is None: return None
    resized_img = image.resize(target_size)
    return resized_img

def normalize_image(image):
    if image is None: return None
    img_array = np.array(image).astype('float32')
    if img_array.ndim == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    normalized_img = img_array / 255.0
    normalized_img = np.expand_dims(normalized_img, axis=0)
    return normalized_img

def cnn_predict(model, preprocessed_image):
    if preprocessed_image is None: return None
    predictions = model.predict(preprocessed_image)
    return predictions

def calculate_confidence(predictions):
    if predictions is None: return None
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    return predicted_class_idx, confidence

def output_label(predicted_class_idx, class_names):
    if predicted_class_idx is None: return "Unknown"
    predicted_label = class_names[predicted_class_idx]
    return predicted_label

# Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and let the CNN classify it into one of 10 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess and predict
    resized_img = resize_image(image)
    preprocessed_img = normalize_image(resized_img)

    if preprocessed_img is not None:
        with st.spinner('Classifying image...'):
            predictions = cnn_predict(model, preprocessed_img)
            predicted_class_idx, confidence = calculate_confidence(predictions)
            predicted_label = output_label(predicted_class_idx, class_names)

        st.success("Classification Complete!")
        st.write(f"Predicted Class: **{predicted_label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("Could not preprocess the image.")
