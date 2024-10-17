import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "nvidia/NVLM-D-72B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

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
