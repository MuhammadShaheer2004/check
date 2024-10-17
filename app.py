import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Set up the model and tokenizer
model_name = "rhymes-ai/Aria"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the pipeline
text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Streamlit app
st.title("Text Generation with Aria")

# Input for user
user_input = st.text_area("Enter your prompt:", "Once upon a time")

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        result = text_generation(user_input, max_length=100, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        st.write("Generated Text:")
        st.write(generated_text)
