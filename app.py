
import streamlit as st
from object_detection.yolo_detection import detect_objects
from image_classification.classify_image import classify_image
from text_generation.generate_caption import generate_caption

st.title("Smart Multimedia AI System")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    st.subheader("1. Object Detection")
    detect_objects("temp.jpg")

    st.subheader("2. Image Classification")
    classes = classify_image("temp.jpg")
    for c in classes:
        st.write(f"{c[1]}: {round(c[2]*100, 2)}%")

    st.subheader("3. Text Generation")
    caption = generate_caption("An image of " + classes[0][1])
    st.write(caption)
