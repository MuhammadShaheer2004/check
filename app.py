# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Hugging_Face": "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"}
# Initialize the image-to-text pipeline
pipe = pipeline("image-to-text", model="paragon-AI/blip2-image-to-text")

# Set up the Streamlit app
st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Generate the caption
    with st.spinner("Generating caption..."):
        result = pipe(image)
        caption = result[0]['generated_text']  # Adjust according to the output structure
        st.write("Generated Caption:")
        st.write(caption)
