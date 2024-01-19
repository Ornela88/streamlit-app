import streamlit as st
import pandas as pd

# Load movie and ratings data (replace these URLs with your own data)
movies_url = 'https://drive.google.com/file/d/1gNuiAz9GPT1zlSTO1M0Yxg5KPHtrUy_E/view?usp=sharing'
ratings_url = 'https://drive.google.com/file/d/1-agUGx3-nXeJKe0SF-sZrcW1R5j3eWOu/view?view?usp=sharing'

# Function to load data from Google Drive URLs
def load_data_from_drive(url):
    path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(path)

# Load data
movies_df = load_data_from_drive(movies_url)
ratings_df = load_data_from_drive(ratings_url)

# Calculate movie popularity by counting the number of ratings
movie_popularity = ratings_df['movieId'].value_counts().reset_index()
movie_popularity.columns = ['movieId', 'popularity']

# Sort movies by popularity
movie_popularity = movie_popularity.sort_values(by='popularity', ascending=False)

# Streamlit app
st.title('Movie Recommender System')

# Display top popular movies
st.header('Most Popular Movies:')
for i, row in enumerate(movie_popularity.head(10).itertuples(), 1):
    movie_id = row.movieId
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if not movie_info.empty:
        title = movie_info.iloc[0]['title']
        st.write(f"{i}. Movie Title: {title}")

