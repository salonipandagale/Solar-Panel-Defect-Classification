import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import base64


# --------------------------------------------------
# Background
# --------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

    st.markdown(bg_css, unsafe_allow_html=True)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Solar Panel Defect Classifier",
    page_icon="☀️",
    layout="centered"
)

add_bg_from_local("background.jpg")


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("☀️ Solar Panel Defect Classifier")

st.write(
    "Upload an image of a solar panel to detect defects "
    "using a deep learning model."
)


# --------------------------------------------------
# Load SavedModel
# --------------------------------------------------
@st.cache_resource
def load_model():

    model = tf.saved_model.load("solar_model_deploy")

    # Get the prediction function
    predict_fn = model.signatures["serving_default"]

    return predict_fn


with st.spinner("Loading model..."):
    predict_fn = load_model()


# --------------------------------------------------
# Classes
# --------------------------------------------------
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered"
]


# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a solar panel image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # ----------------------------------------------
    # Read image
    # ----------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )


    # ----------------------------------------------
    # Preprocess image
    # ----------------------------------------------
    img = image.resize((224, 224))

    img_array = np.array(img).astype(np.float32)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------
    with st.spinner("Analyzing the panel..."):

        predictions_dict = predict_fn(
            random_flip_input=tf.convert_to_tensor(
                img_array,
                dtype=tf.float32
            )
        )

        predictions = predictions_dict["dense_1"].numpy()[0]


    # ----------------------------------------------
    # Predicted class
    # ----------------------------------------------
    predicted_idx = int(np.argmax(predictions))

    confidence = float(predictions[predicted_idx])

    predicted_class = CLASSES[predicted_idx]


    # ----------------------------------------------
    # Main prediction
    # ----------------------------------------------
    st.markdown(
        f"""
        ### **Prediction: {predicted_class}**

        **Confidence: {confidence:.1%}**
        """
    )


    # ----------------------------------------------
    # Condition message
    # ----------------------------------------------
    if predicted_class == "Clean":

        st.success(
            "The panel appears to be in good condition!"
        )

    else:

        st.warning(
            "A defect or contamination has been detected!"
        )


    # ----------------------------------------------
    # Top 3 predictions
    # ----------------------------------------------
    st.write("### Top 3 Predictions")

    top_indices = np.argsort(predictions)[-3:][::-1]


    for i, idx in enumerate(top_indices):

        class_name = CLASSES[idx]

        prob = predictions[idx]


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


    # ----------------------------------------------
    # All probabilities
    # ----------------------------------------------
    with st.expander("View all class probabilities"):

        for i, prob in enumerate(predictions):

            st.write(
                f"{CLASSES[i]:<20} {prob:.1%}"
            )