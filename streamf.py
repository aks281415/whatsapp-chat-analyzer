import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Assuming 'preprocess' and 'prepare_data_for_model' are defined in preprocess.py
from preprocessing import preprocess
import re
# Load the model
model_path = 'E:/whatsapp-chat-analyzer/IMDB_sentiment_analysis0.87'
model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')  # Assuming the model can be loaded like this for simplicity




def preprocess(data):
    # Adjust pattern to include AM/PM and handle potential non-breaking spaces or special characters
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date type with error handling, adjusting to 12-hour clock with AM/PM
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ')
    except ValueError:
        print("Date format error: Ensure your date format matches the pattern specified.")

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract user and message
    df[['user', 'message']] = df['user_message'].str.extract('(?:(.+?):\s)?(.*)')
    df['user'].fillna('group_notification', inplace=True)
    df.drop(columns=['user_message'], inplace=True)

    # Extract date and time components
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.strftime('%B')
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.strftime('%A')
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df


# Prepare text data for the model
def prepare_data_for_model(texts, max_len=500):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

def predict_sentiments(df):
    # Exclude 'group_notification' from analysis
    df = df[df['user'] != 'group_notification']

    messages = df['message'].tolist()
    prepared_data = prepare_data_for_model(messages)
    predictions = model(prepared_data)

    # Assuming 'dense' is the correct output key based on your model's architecture
    if 'dense' in predictions:
        predictions_tensor = predictions['dense']
    else:
        raise KeyError("Expected key 'dense' not found in model predictions. Available keys: " + str(predictions.keys()))

    predictions_array = predictions_tensor.numpy()  # Convert the tensor to a numpy array
    predictions_list = predictions_array.flatten().tolist()  # Flatten the array and convert to list
    df['sentiment_score'] = predictions_list

    return df


def display_overall_sentiments(df):
    # Calculate the overall average sentiment score
    overall_average = np.mean(df['sentiment_score'])

    # Convert the overall average to an emoji based on predefined thresholds
    if overall_average > 0.85:
        overall_sentiment = "Positive 🙂"  # Positive
    elif overall_average > 0.55:
        overall_sentiment = "Neutral 😐"  # Neutral
    else:
        overall_sentiment = "Negative 🙁"  # Negative

    # Display the overall average sentiment score and corresponding emoji
    st.write(f"Overall Average Sentiment: {overall_sentiment}")


def display_user_sentiments(df):
    user_sentiment_average = df.groupby('user')['sentiment_score'].mean().round(2)

    # Apply the emoji sentiment conversion directly within the same function
    user_sentiment_emojis = user_sentiment_average.apply(lambda score: "Positive 🙂" if score > 0.85 else "Neutral 😐" if score > 0.55 else "Negative 🙁")

    # Create a DataFrame to display
    sentiment_display = pd.DataFrame({
        'Sentiment': user_sentiment_emojis
    })

    st.write("Average Sentiment Score per User:")
    st.dataframe(sentiment_display)

    most_positive_users = user_sentiment_average[user_sentiment_average >= 0.92]
    if not most_positive_users.empty:
        st.write("Users with overwhelmingly positive sentiment 🙂:", ', '.join(most_positive_users.index.tolist()))
    else:
        st.write("No users with overwhelmingly positive sentiment.")



def main():
    st.title('WhatsApp Chat Sentiment Analyzer')
    uploaded_file = st.file_uploader("Choose a chat file", type=['txt', 'csv'], key='file_uploader')

    if uploaded_file is not None:
        chat_data = uploaded_file.getvalue().decode("utf-8")  # Decode the uploaded file content
        df = preprocess(chat_data)  # Process the chat data

        # Center the "Analyze Sentiments" button using columns
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
                
                if st.button('Show Sentiment Per User', key='per_user_button'):
                    display_user_sentiments(st.session_state['result_df'])

if __name__ == '__main__':
    main()



