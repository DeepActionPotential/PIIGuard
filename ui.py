import streamlit as st
from utils import prepare_inputs
import torch
import pandas as pd

def render_ui(text, model, tokenizer, idx2tag):
    # Prepare inputs
    input_ids, mask = prepare_inputs(text, tokenizer)

    # Run model
    with torch.no_grad():
        predictions = model(input_ids, mask=mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [idx2tag.get(tag, "O") for tag in predictions[0]]

    # Build table data
    rows = []
    for token, label in zip(tokens, labels):
        rows.append({"Token": token, "Predicted Label": label})

    df = pd.DataFrame(rows)

    # Show in Streamlit
    st.subheader("🔍 Predictions")
    st.dataframe(df, use_container_width=True)  # or st.table(df) for static table
