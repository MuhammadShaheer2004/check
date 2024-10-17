import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Hugging Face token
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

# Load model and tokenizer
model_name = "nvidia/NVLM-D-72B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    st.write("Model and tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit interface
st.title("NVLM Model Interface")
user_input = st.text_input("Enter your text prompt:")

if st.button("Generate Response"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")

