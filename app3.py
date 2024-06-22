import pandas as pd
import numpy as np
import tensorflow as tf
import re

from tensorflow import keras
from preprocessing import preprocess

# Correctly load the trained sentiment analysis model using TensorFlow's Keras API
model_path = r'E:\whatsapp-chat-analyzer\IMDB_sentiment_analysis0.87'  # Use raw string for path
model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')  # Correct model loading for direct predictions

# Load and preprocess the chat data
def load_and_preprocess_chat_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chat_data = file.read()
    return preprocess(chat_data)

# Prepare text data for the model
def prepare_data_for_model(texts, max_len=500):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)


def predict_sentiments(df):
    # Replace specific names
    df['user'] = df['user'].replace({
        'Sandeep Jha (Not Alien)': 'Sandeep Jha',
        'Ankit Patla 2': 'Ankit'
    })

    # Filter out group_notification before processing
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
    df['sentiment_score'] = predictions_list  # Add predictions back to the DataFrame

    user_input = input("Type 'overall' for overall sentiment score, or 'per user' for individual user scores: ").strip().lower()

    if user_input == 'overall':
        overall_average = np.mean(df['sentiment_score'])
        print(f"Overall Average Sentiment Score: {overall_average:.2f}")
    elif user_input == 'per user':
        # Calculate and display sentiment scores per user
        user_sentiment_average = df.groupby('user')['sentiment_score'].mean().round(2).to_dict()
        print("Average Sentiment Score per User:")
        for user, score in user_sentiment_average.items():
            print(f"{user}: {score:.2f}")
        
        # Identify the most positive users
        most_positive_users = [user for user, score in user_sentiment_average.items() if score >= 0.92]
        if most_positive_users:
            print("Users with overwhelmingly positive sentiment:", ', '.join(most_positive_users))
        else:
            print("No users with overwhelmingly positive sentiment.")
    else:
        print("Invalid input. Please enter 'overall' or 'per user'.")

    return user_sentiment_average






# Main function to run the analysis
def main():
    file_path = r'E:\whatsapp-chat-analyzer\chat.txt'
    df = load_and_preprocess_chat_data(file_path)
    predictions = predict_sentiments(df)

if __name__ == "__main__":
    main()
