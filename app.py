import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title('🐾 Cat vs. Dog Image Classifier')
img_size = 256  # MobileNetV2 training size
PRETRAINED_MODEL_PATH = "model_transfer.keras"
CATEGORIES = ["Cat", "Dog"]  # Alphabetical order: cat=0, dog=1

# Load the model with caching so it doesn't reload on every button click
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(PRETRAINED_MODEL_PATH)

try:
    model = load_model()
    print('Model Loaded Successfully')
except Exception as e:
    st.error(f"Error loading model file: {e}")

def load_classifier():
    st.subheader("Upload an image of a cat or a dog to check the prediction")
    file = st.file_uploader(label=" ", type=['jpeg', 'jpg', 'png'])

    if file is not None:
        # Load and resize image using Keras preprocessing utilities
        img = tf.keras.preprocessing.image.load_img(file, target_size=(img_size, img_size))
        new_array = tf.keras.preprocessing.image.img_to_array(img)
        new_array = new_array.reshape(-1, img_size, img_size, 3)
        
        # Updated parameter to avoid the warning
        st.image(file, caption="Uploaded Image", use_container_width=True)
        st.write("")
        st.write("")
            
        if st.button("PREDICT"):
            with st.spinner("Analyzing..."):
                # Making prediction
                # Dividing by 255.0 puts inputs in [0, 1], which your model's 
                # internal Rescaling layer then shifts to [-1, 1]
                prediction = model.predict(new_array / 255.0)
                print(prediction)
                
                score = prediction[0][0]
                print(score)
                
                # Determine category and confidence
                class_idx = 0 if score < 0.5 else 1
                confidence = (1 - score) * 100 if class_idx == 0 else score * 100
                
                preds = f"{CATEGORIES[class_idx]} - {confidence:.2f}% (Raw Score: {score:.4f})"
                
                # Display result cleanly
                if class_idx == 0:
                    st.success(f"### Prediction: 🐱 {preds}")
                else:
                    st.success(f"### Prediction: 🐶 {preds}")

def main():
    load_classifier()

if __name__ == "__main__":
    main()