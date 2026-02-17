import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "arjken/FinancialTweetSentimentAnalyzer"  # or local path like "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Financial Tweet Sentiment Analyzer")

tweet = st.text_area("Enter a financial tweet:")

if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter some text first.")
    else:
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        st.write(f"Predicted sentiment class: {predicted_class}")