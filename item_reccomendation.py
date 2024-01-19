import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movie data (replace this URL with your own data)
movies_url = 'https://drive.google.com/file/d/1gNuiAz9GPT1zlSTO1M0Yxg5KPHtrUy_E/view?usp=sharing'

# Function to load data from Google Drive URLs
def load_data_from_drive(url):
    path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(path)

# Load movie data
movies_df = load_data_from_drive(movies_url)

# Create a TF-IDF vectorizer to convert movie genres into numerical data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Calculate the cosine similarity between movies based on genres
cosine_similarity_scores = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a DataFrame for movie similarity scores
movie_similarity_df = pd.DataFrame(cosine_similarity_scores, index=movies_df['movieId'], columns=movies_df['movieId'])

# Set app title and header style
st.title('Item-Based Movie Recommender System')

# Sidebar for user input
movie_title_to_recommend = st.sidebar.text_input('Enter Movie Title:', 'Toy Story (1995)')

# Function to get movie ID based on the movie title
def get_movie_id(movie_title):
    movie = movies_df[movies_df['title'] == movie_title]
    if not movie.empty:
        return movie.iloc[0]['movieId']
    else:
        return None

movie_id_to_recommend = get_movie_id(movie_title_to_recommend)

# Generate movie recommendations based on item similarity
movie_recommendations = []

if movie_id_to_recommend is not None and st.sidebar.button('Recommend Movies'):
    similar_movies = movie_similarity_df[movie_id_to_recommend].sort_values(ascending=False)[1:11]
    movie_recommendations = similar_movies.index.tolist()

# Display top movie recommendations with movie titles
st.sidebar.header('Top Movie Recommendations (Item-Based):')
for i, movie_id in enumerate(movie_recommendations, 1):
    movie_info = movies_df[movies_df['movieId'] == movie_id]
    if not movie_info.empty:
        title = movie_info.iloc[0]['title']
        st.sidebar.write(f"{i}. Movie Title: {title}")

# Display selected movie information
selected_movie_info = movies_df[movies_df['movieId'] == movie_id_to_recommend]

if not selected_movie_info.empty:
    st.header('Selected Movie Information:')
    st.write(f"Movie Title: {selected_movie_info.iloc[0]['title']}")
    st.write(f"Genres: {selected_movie_info.iloc[0]['genres']}")

