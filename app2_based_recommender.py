import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

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

# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train an SVD recommender on the training set
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
svd.fit(trainset)

# Apply custom CSS style
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px #888888;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px #888888;
    }
    .sidebar .sidebar-content .markdown-text-container {
        background-color: #e0e0e0;
    }
    .stButton>button {
        background-color: #0073e6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stTextInput>div>input {
        background-color: #ffffff;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Set app title and header style
st.title('Movie Recommender System')

# Sidebar for user input
user_id_to_recommend = st.sidebar.number_input('Enter User ID:', min_value=1, max_value=ratings_df['userId'].max(), value=1)

# Generate movie recommendations for the selected user
user_movies = ratings_df[ratings_df['userId'] == user_id_to_recommend]['movieId'].unique()
movies_to_recommend = [movie_id for movie_id in ratings_df['movieId'].unique() if movie_id not in user_movies]
user_recommendations = []

if st.button('Recommend Movies'):
    for movie_id in movies_to_recommend:
        estimated_rating = svd.predict(user_id_to_recommend, movie_id).est
        user_recommendations.append((movie_id, estimated_rating))

    user_recommendations.sort(key=lambda x: x[1], reverse=True)

    # Display top movie recommendations with movie titles
    st.header('Top Movie Recommendations:')
    for i, (movie_id, estimated_rating) in enumerate(user_recommendations[:10], 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            st.write(f"{i}. Movie Title: {title}")
            st.write(f"   Estimated Rating: {estimated_rating:.2f}")

# Function to get top-rated movies for a user
def get_top_rated_movies(user_id, n):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    top_rated_movies = user_ratings.nlargest(n, 'rating')
    
    # Join with movies_df to get movie titles
    top_rated_movies_with_titles = top_rated_movies.merge(movies_df, on='movieId', how='left')
    
    return top_rated_movies_with_titles

# Display top-rated movies for the selected user
top_rated_movies_df = get_top_rated_movies(user_id_to_recommend, 5)

if not top_rated_movies_df.empty:
    st.sidebar.header('User Top-Rated Movies:')
    for i, row in enumerate(top_rated_movies_df.iterrows(), 1):
        movie_title = row[1]['title']
        rating = row[1]['rating']
        st.sidebar.write(f"{i}. Movie Title: {movie_title}")
        st.sidebar.write(f"   Rating: {rating:.2f}")

