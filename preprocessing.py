import re
import pandas as pd

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

    # Define periods more intuitively
    df['period'] = df['hour'].apply(lambda x: f"{x:02d}-{(x+1)%24:02d}")

    return df


