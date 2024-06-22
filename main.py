from onechat import chat_analyzer
from twosenti import run_senti

import streamlit as st
  # Import the new function from senti.py

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Chat Analyzer", "Sentiment Analysis"])

    if page == "Home":
        st.subheader("Welcome to the WhatsApp Chat and Sentiment Analyzer")
        st.write("Select a feature from the left sidebar to get started.")
    
    elif page == "Chat Analyzer":
        # Assume chat_analyzer function is defined in a separate module or here
        chat_analyzer()  
    
    elif page == "Sentiment Analysis":
        run_senti()  # Run the entire sentiment analysis functionality

if __name__ == '__main__':
    main()
