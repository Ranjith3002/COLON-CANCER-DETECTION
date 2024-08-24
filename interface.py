import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax

# Load the pre-trained model for Streamlit prediction
loaded_model = tf.keras.models.load_model("F:\\Colon Cancer.h5", compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define classes for the prediction
class_labels = ['Cancerous', 'Non_Cancerous']

# Function to preprocess the image
def preprocess_image_streamlit(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Function to make prediction
def predict_streamlit(image_array):
    predictions = loaded_model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    return class_labels[tf.argmax(score)]

# Streamlit app
st.title('Colon Cancer Prediction')

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when predict button is clicked
    if st.button('Predict'):
        img_array = preprocess_image_streamlit(image)
        prediction = predict_streamlit(img_array)
        st.success(f'The image is predicted as: {prediction}')
