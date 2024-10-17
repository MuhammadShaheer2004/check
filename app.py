import streamlit as st
from transformers import NVLM_D
import torch
import os

# Load the NVLM-D model with the Hugging Face token
model_name = "nvidia/NVLM-D-72B"
hf_token = os.getenv("HF_TOKEN")  # Get the token from environment variables

model = NVLM_D.from_pretrained(model_name, trust_remote_code=True, use_auth_token=hf_token)

# Streamlit app title
st.title("Text Generation with NVLM-D")

# User input
user_input = st.text_area("Enter your prompt:", "Once upon a time")

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        # Generate text
        inputs = model.tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated Text:")
        st.write(generated_text)
