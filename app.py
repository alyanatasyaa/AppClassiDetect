import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

@st.cache_resource
def load_models():
    detection_model = YOLO("model/AlyaNatasya_Laporan4.pt")
    classifier_model = tf.keras.models.load_model("model/AlyaNatasya_Laporan2.h5")
    return detection_model, classifier_model

detection_model, classifier_model = load_models()

def classify_image(img, st):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = classifier_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    class_labels = {
        0: "Garbage Bag",
        1: "Paper Bag",
        2: "Plastic Bag"
    }

    st.subheader("Classification Result")
    st.success(f"Prediction: **{class_labels[predicted_class]}** ({confidence:.2f}% confidence)")

def detect_objects(img, st):
    results = detection_model(img)
    result = results[0]
    im_array = result.plot()

    st.subheader("Detection Result")
    st.success(f"Number of objects detected: **{len(result.boxes)}**")
    st.image(im_array[..., ::-1], channels="RGB", use_container_width=True)

st.set_page_config( page_title="ClassiDetect", page_icon="⚙️")

col1, col2 = st.sidebar.columns([0.5, 3])
with col1:
    dark_mode = st.toggle("", value=False)
with col2:
    st.markdown("### 🌙 Dark Mode")

if dark_mode:
    primary_bg = "#11121E"
    text_color = "white"
    sidebar_bg = "#1D1D29"
    button_bg = "#11121E"
    border_color = "#444444"
    toggle_active = "#F8D227"
else:
    primary_bg = "#F3F3F3"
    text_color = "black"
    sidebar_bg = "#FFFFFF"
    button_bg = "#F3F3F3"
    border_color = "#B9B5B5"
    toggle_active = "#B9B5B5"

def set_custom_css(primary_bg, sidebar_bg, text_color, 
                   border_color, button_bg, toggle_active):
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"],
        [data-testid="stToolbar"],
        [data-testid="stFullScreenFrame"],
        [data-testid="stMainMenuList"] li {{
            background-color: {primary_bg} !important;
            transition: all 0.4s ease-in-out;
        }}

        [data-testid="stSidebar"], 
        [data-testid="stSidebarContent"] {{
            background-color: {sidebar_bg} !important;
            height: 100vh !important;
            min-height: 100vh !important;
        }}

        [data-baseweb="select"] > div,
        [data-testid="stSelectboxVirtualDropdown"] {{
            background-color: {primary_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            transition: all 0.4s ease-in-out;
        }}

        [data-baseweb="popover"] > div {{
            background-color: {primary_bg} !important;
        }}

        [data-baseweb="icon"] {{
            fill: {text_color} !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: {primary_bg} !important;
            border: 1px dashed {border_color} !important;
            border-radius: 8px !important;
            transition: all 0.3s ease-in-out;
        }}

        [data-testid="stBaseButton-secondary"],
        [data-testid="stCameraInputButton"] {{
            background-color: {sidebar_bg} !important;
            border: 1px solid {border_color} !important;
        }}

        [data-testid="stElementToolbarButtonContainer"] {{
            background-color: {button_bg} !important;
        }}

        [data-baseweb="checkbox"] > div {{
            background-color: {toggle_active} !important;
        }}

        hr {{
            background-color: {border_color} !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: {text_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

set_custom_css(primary_bg, sidebar_bg, text_color, border_color, button_bg, toggle_active)

def main():
    
    st.sidebar.title("Settings")
       
    app_mode = st.sidebar.selectbox(
        'Choose the App Mode', 
        ['About App', 'Image Classification', 'Object Detection']
        )
    
    
    if app_mode == 'About App':
        
        st.header("Welcome to ClassiDetect! ⚙️")
                
        st.markdown(
        """
        <p>This dashboard showcases two exciting <b>Computer Vision</b> tasks powered by <b>Deep Learning</b> models:</p>
        <p>🖼️ <b>Image Classification</b> using a <b>Convolutional Neural Network (CNN)</b><br>
        🔎 <b>Object Detection</b> using <b>YOLOv8n</b></p>

        <p>The goal of this dashboard is to make it easier to understand how these models work,
        visualize their results, and interactively test them with your own images.</p>

        <hr>

        <h4>🚀 How to Use the App:</h4>
        <p>
        &#x2022; In the sidebar, select the desired mode under <b>“Choose the App Mode”</b>.<br>
        &#x2022; Choose <b>“Image Classification”</b> to upload an image and let the CNN model predict its category.<br>
        &#x2022; Choose <b>“Object Detection”</b> to upload an image and detect object using YOLOv8n.<br>
        &#x2022; You can also switch between <b>Light</b> and <b>Dark Mode</b> using the toggle in the sidebar.<br>
        &#x2022; The app will automatically process and display results, including detected objects and class labels.
        </p>

        <hr>

        <p><b>Enjoy exploring the app and have fun experimenting with computer vision!</b></p>
        """, unsafe_allow_html=True
    )
    
    elif app_mode == "Image Classification":
        
        st.header("🖼️ Classification Image with CNN")
        
        st.sidebar.markdown("----")
        
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'])
        capture_image = st.sidebar.camera_input("Or take a photo using webcam")
        st.sidebar.warning(
        "This model is optimized to classify garbage, paper, and plastic bags only. The model may not perform accurately on other objects.",
        icon=":material/info:",
        )
        
        DEMO_CLASS_IMAGE = "sample_images/00001348.jpg"

        if capture_image is not None:
            image = np.array(Image.open(capture_image))
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            image = np.array(Image.open(uploaded_file))
        else:
            DEMO_CLASS_IMAGE = "sample_images/00001348.jpg"
            img = cv2.imread(DEMO_CLASS_IMAGE)
            image = np.array(Image.open(DEMO_CLASS_IMAGE))

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        classify_image(img, st)
    
    elif app_mode == "Object Detection":
        
        st.header("🔎 Object Detection with YOLOv8n",)
        
        st.sidebar.markdown("----")
                
        uploaded_file  = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'])
        capture_image = st.sidebar.camera_input("Or take a photo using webcam")
        
        st.sidebar.warning(
        "This model is optimized to detect only penguin and turtle.",
        icon=":material/info:",
        )
        
        DEMO_DETECT_IMAGE = "sample_images/image_id_118.jpg"
        
        if capture_image is not None:
            image = np.array(Image.open(capture_image))
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            image = np.array(Image.open(uploaded_file))
        else:
            img = cv2.imread(DEMO_DETECT_IMAGE)
            image = np.array(Image.open(DEMO_DETECT_IMAGE))
            
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        detect_objects(img, st)     
    
if __name__ == "__main__":
    try:              
        main()
    except SystemExit:
        pass
        