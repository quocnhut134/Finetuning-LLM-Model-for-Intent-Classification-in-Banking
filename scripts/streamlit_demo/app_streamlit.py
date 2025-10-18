import streamlit as st
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import torch
import os

def load_css(file_name):
    current_dir = os.path.dirname(__file__)
    css_file_path = os.path.join(current_dir, file_name)
    with open(css_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

st.set_page_config(
    page_title="Intent Classification in Banking",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_model_and_mapping():
    model_path = "saved_models/distilbert_distilbert-base-uncased-finetuned-banking77"
    
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_path, device=device)
    
    raw_datasets = load_dataset("mteb/banking77")
    df_train = raw_datasets['train'].to_pandas()
    
    label_mapping_df = df_train[['label', 'label_text']].drop_duplicates().reset_index(drop=True)
    
    return classifier, label_mapping_df

classifier, label_mapping_df = load_model_and_mapping()

st.title("Intent Classification in Banking")

st.markdown(
    """
    This is a demo for a language model that has been fine-tuned to classify 77 different types of customer requests in the banking sector.
    """
)

with st.form("intent_form"):
    user_input = st.text_area("Please enter your request here:", "", height=100)
    submitted = st.form_submit_button("Classify Intent")

if submitted and user_input:
    with st.spinner('The model is analyzing...'):
        predictions = classifier(user_input, top_k=5)
        
        mapped_predictions = []
        for pred in predictions:
            label_id = int(pred['label'].split('_')[1])
            
            matched_row = label_mapping_df[label_mapping_df['label'] == label_id]
            
            if not matched_row.empty:
                label_text = matched_row['label_text'].iloc[0]
            else:
                label_text = "Cannot find label"
            
            mapped_predictions.append({'label': label_text, 'score': pred['score']})

        top_prediction = mapped_predictions[0]
        
        st.success(f"**The top predicted intent is:** `{top_prediction['label']}`")
        st.metric(label="With confidence", value=f"{top_prediction['score']:.2%}")
        
        st.markdown("---")
        
        st.subheader("Other possibilities:")
        
        df = pd.DataFrame(mapped_predictions)
        df = df.rename(columns={'label': 'Intent', 'score': 'Confidence'})
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2%}")

        st.dataframe(
            df,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=True,
            width='stretch'
        )

elif submitted and not user_input:
    st.warning("Please enter a request to classify.")