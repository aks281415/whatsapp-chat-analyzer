import streamlit as st
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pandas as pd
# from app import perform_sentiment_analysis

from tensorflow import keras
from keras import layers,models
#from keras.models import load_model
import tensorflow as tf
from preprocessing import preprocess



def chat_analyzer():
    st.sidebar.title("WhatsApp Chat Analyzer")
    uploaded_file = st.sidebar.file_uploader("Choose a chat file", type=['txt', 'csv'])

    if uploaded_file is not None:
        data = uploaded_file.getvalue().decode("utf-8")
        df = preprocess(data)

        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

        if st.sidebar.button("Show Analysis"):
            st.title('WhatsApp Chat Analyzer')
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            st.title("Top Statistics of your chat")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # Sentiment Analysis Display
            # st.title("Sentiment Analysis")
            # if selected_user != "Overall":
            #     polarity = df[df['user'] == selected_user]['polarity'].mean()
            #     subjectivity = df[df['user'] == selected_user]['subjectivity'].mean()
            # else:
            #     polarity = df['polarity'].mean()
            #     subjectivity = df['subjectivity'].mean()

            # # Display the numerical values
            # st.write(f"Average sentiment polarity: {polarity:.2f}")
            # st.write(f"Average sentiment subjectivity: {subjectivity:.2f}")

            # # Adding interpretive text for polarity
            # if polarity > 0:
            #     if polarity < 0.3:
            #         polarity_text = "The text is slightly positive."
            #     elif polarity < 0.6:
            #         polarity_text = "The text shows positivity."
            #     else:
            #         polarity_text = "The text is highly positive!"
            # elif polarity < 0:
            #     if polarity > -0.3:
            #         polarity_text = "The text is slightly negative."
            #     elif polarity > -0.6:
            #         polarity_text = "The text shows negativity."
            #     else:
            #         polarity_text = "The text is highly negative!"
            # else:
            #     polarity_text = "The text is neutral."

            # # Adding interpretive text for subjectivity
            # if subjectivity < 0.3:
            #     subjectivity_text = "The text is mostly factual."
            # elif subjectivity < 0.6:
            #     subjectivity_text = "The text is somewhat opinionated."
            # else:
            #     subjectivity_text = "The text is highly opinionated."

            # # Display the explanatory text
            # st.write(polarity_text)
            # st.write(subjectivity_text)

            # Topic Modeling
            st.title("Identified Topics")
            topics = helper.identify_topics(df['message'].tolist())
            for i, topic in enumerate(topics):
                st.write(f"Topic {i+1}: {', '.join(topic)}")

            # Monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)
            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Weekly Activity Map
            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # Finding the busiest users in the group (Group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()
                col1, col2 = st.columns(2)
                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # Most common words
            most_common_df = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.title('Most common words')
            st.pyplot(fig)

            # Emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)