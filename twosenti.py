import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import re
from streamapp2 import preprocess, prepare_data_for_model,predict_sentiments,display_overall_sentiments,display_user_sentiments

def run_senti():
    st.title('WhatsApp Chat Sentiment Analyzer')
    uploaded_file = st.file_uploader("Choose a chat file", type=['txt', 'csv'], key='file_uploader')

    if uploaded_file is not None:
        chat_data = uploaded_file.getvalue().decode("utf-8")  # Decode the uploaded file content
        df = preprocess(chat_data)  # Process the chat data

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button('Analyze Sentiments', key='analyze_button'):
                # Perform sentiment analysis when this button is clicked
                result_df = predict_sentiments(df)
                st.session_state['result_df'] = result_df  # Store results in session state
                st.write("Sentiment Analysis Completed. Choose an analysis type below.")

        if 'result_df' in st.session_state:
            # Create columns to center the result display buttons
            col1, col2, col3 = st.columns([1,2,1])
            with col2:  # Place buttons in the center column
                if st.button('Show Overall Sentiment', key='overall_button'):
                    display_overall_sentiments(st.session_state['result_df'])
                
                if st.button('Sentimental analysis for each user', key='per_user_button'):
                    display_user_sentiments(st.session_state['result_df'])