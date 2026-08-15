import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import base64

st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")

st.title("☀️ Solar Panel Defect Classifier")

st.write(
    "Upload an image of a solar panel to detect defects "
    "using a deep learning model."
)

CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered"
]

uploaded_file = st.file_uploader(
    "Upload a solar panel image...",
    type=["jpg", "jpeg", "png"]
)

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.saved_model.load("solar_model_deploy")
    return model.signatures["serving_default"]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Loading AI model..."):
        predict_fn = load_model()

    img = image.resize((224, 224))

    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing the panel..."):
        predictions_dict = predict_fn(
            random_flip_input=tf.convert_to_tensor(
                img_array,
                dtype=tf.float32
            )
        )

        predictions = predictions_dict["dense_1"].numpy()[0]

    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])
    predicted_class = CLASSES[predicted_idx]

    st.markdown(
        f"""
        ### Prediction: {predicted_class}

        **Confidence: {confidence:.1%}**
        """
    )

    if predicted_class == "Clean":
        st.success("The panel appears to be in good condition!")
    else:
        st.warning("A defect or contamination has been detected!")

    st.write("### Top 3 Predictions")

    top_indices = np.argsort(predictions)[-3:][::-1]

    for i, idx in enumerate(top_indices):
        class_name = CLASSES[idx]
        prob = float(predictions[idx])

        if i == 0:
            st.markdown(
                f"**1st**: **{class_name}** - {prob:.1%}"
            )
        elif i == 1:
            st.markdown(
                f"**2nd**: {class_name} - {prob:.1%}"
            )
        else:
            st.markdown(
                f"**3rd**: {class_name} - {prob:.1%}"
            )

    with st.expander("View all class probabilities"):
        for i, prob in enumerate(predictions):
            st.write(
                f"{CLASSES[i]}: {float(prob):.1%}"
            )
