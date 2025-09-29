import streamlit as st
from utils import load_full_model_and_tokenizer
from ui import render_ui
from model import BiLSTMCRF

# Cache model and tokenizer
@st.cache_resource
def get_model_and_tokenizer():
    return load_full_model_and_tokenizer("models/best_bilstm_crf_model.pt")

model, tokenizer, idx2tag = get_model_and_tokenizer()

def main():
    st.title("🔒 Detecting PII with BiLSTM-CRF")

    text = st.text_area("Enter text to analyze:", height=200)

    if st.button("Analyze"):
        if text.strip():
            render_ui(text, model, tokenizer, idx2tag)
        else:
            st.warning("⚠️ Please enter some text.")

if __name__ == "__main__":
    main()
