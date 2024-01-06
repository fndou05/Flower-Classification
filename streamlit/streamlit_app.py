# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page
st.set_page_config(page_title="BlossomBotanist", page_icon=":cherry_blossom:", initial_sidebar_state='auto')

# hide the part of the code for custom CSS styling
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define the model loading function
@st.cache_resource
def load_model():
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)

    # Build the path to the model file
    model_path = os.path.join(current_dir, 'models', 'VGG-16.h5')

    # Load the model
    model = tf.keras.models.load_model(model_path)

    return model

# Load the model at the beginning of the app
model = load_model()

with st.sidebar:
    st.image('../images/streamlit-image.jpg')
    st.title("BlossomBotanist")
    st.subheader("Image Based Florapedia")

st.write("# Floral Classification")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128,128)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    # Debugging: Print the predictions array
    st.write("Raw model predictions:", predictions)

    max_confidence = np.max(predictions)
    st.sidebar.error(f"Confidence: {max_confidence * 100:.2f}%")

    # Check if confidence is below 50%
    if max_confidence < 0.5:
        st.warning("⚠️ Confidence is low! Please ensure the image is of one of the 14 flower classes our model is trained on.")
    else:
        # Existing classification code
        class_names = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy', 'carnation', 'common_daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip', 'water_lily']
        string = "Detected Flower : " + class_names[np.argmax(predictions)]

    if class_names[np.argmax(predictions)] == 'astilbe':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Astilbe**: Charming, shade-tolerant flowers with fern-like foliage.  \n• **Care**: Thrives in rich, moist soil; prefers partial to full shade.")

    elif class_names[np.argmax(predictions)] == 'bellflower':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Bellflower**: Bell-shaped, usually blue flowers.  \n• **Care**: Prefers well-drained soil; full sun to partial shade.")

    elif class_names[np.argmax(predictions)] == 'black_eyed_susan':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Black-Eyed Susan**: Bright yellow petals with dark centers.  \n• **Care**: Tolerates various soils; best in full sun.")

    elif class_names[np.argmax(predictions)] == 'calendula':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Calendula**: Cheerful orange and yellow flowers with edible petals.  \n• **Care**: Full sun and moderate water; prefers well-drained soil.")

    elif class_names[np.argmax(predictions)] == 'california_poppy':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **California Poppy**: Drought-tolerant wildflowers with silky blooms.  \n• **Care**: Best in full sun; well-drained soil.")

    elif class_names[np.argmax(predictions)] == 'carnation':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Carnation**: Popular for ruffled petals and a wide color range.  \n• **Care**: Needs well-drained, fertile soil; full to part sun.")

    elif class_names[np.argmax(predictions)] == 'common_daisy':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Common Daisy**: Classic white petal and yellow center.  \n• **Care**: Adaptable to most soils; prefers full sun to partial shade.")

    elif class_names[np.argmax(predictions)] == 'coreopsis':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Coreopsis**: Vibrant, daisy-like flowers; easy-care.  \n• **Care**: Loves full sun; tolerates various soil types.")

    elif class_names[np.argmax(predictions)] == 'dandelion':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Dandelion**: Bright yellow flower, turns into a fluffy seed head.  \n• **Care**: Thrives in most conditions; prefers full sun.")

    elif class_names[np.argmax(predictions)] == 'iris':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Iris**: Striking flowers and sword-like foliage; many colors.  \n• **Care**: Requires well-drained soil; full sun.")

    elif class_names[np.argmax(predictions)] == 'rose':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Rose**: Beloved for beauty and fragrance; many varieties.  \n• **Care**: Prefers well-drained, fertile soil; full sun.")

    elif class_names[np.argmax(predictions)] == 'sunflower':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Sunflower**: Large, sunny blooms that follow the sun.  \n• **Care**: Loves full sun; well-drained soil.")

    elif class_names[np.argmax(predictions)] == 'tulip':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Tulip**: One of the first spring blooms; range of colors.  \n• **Care**: Plant bulbs in fall; well-drained soil; full to partial sun.")

    elif class_names[np.argmax(predictions)] == 'water_lily':
        st.sidebar.warning(string)
        st.markdown("## Description")
        st.info("• **Water Lily**: Aquatic plants with floating leaves; star-shaped flowers.  \n• **Care**: Grow in still, shallow water; full sun.")
