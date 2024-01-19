import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import datetime

# Function to load data from Google Drive URLs
def load_data_from_drive(url):
    path = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(path)

# Load movie and ratings data (replace these URLs with your own data)
movies_url = 'https://drive.google.com/file/d/1gNuiAz9GPT1zlSTO1M0Yxg5KPHtrUy_E/view?usp=sharing'
ratings_url = 'https://drive.google.com/file/d/1-agUGx3-nXeJKe0SF-sZrcW1R5j3eWOu/view?usp=sharing'

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

# Create a TF-IDF vectorizer to convert movie genres into numerical data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Calculate the cosine similarity between movies based on genres
cosine_similarity_scores = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a DataFrame for movie similarity scores
movie_similarity_df = pd.DataFrame(cosine_similarity_scores, index=movies_df['movieId'], columns=movies_df['movieId'])

# Set app title and header style
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)

# Header
st.title('Welcome to the Movie Recommender System')
st.write("Discover personalized movie recommendations based on your preferences.")

# Sidebar for user input
st.sidebar.title('Recommendation Type')
recommendation_type = st.sidebar.radio("Select Recommendation Type", ("Popularity-Based", "User-Based", "Item-Based"))

# Sidebar for user input
st.sidebar.title('User Input')

# Popularity-Based Recommendation
if recommendation_type == "Popularity-Based":
    st.sidebar.header('Most Popular Movies:')
    movie_popularity = ratings_df['movieId'].value_counts().reset_index()
    movie_popularity.columns = ['movieId', 'popularity']
    movie_popularity = movie_popularity.sort_values(by='popularity', ascending=False)

    st.sidebar.subheader('Select a Movie:')
    selected_movie_title = st.sidebar.selectbox('', movies_df['title'])
    st.sidebar.subheader('')
    
    def get_movie_id(movie_title):
        movie = movies_df[movies_df['title'] == movie_title]
        if not movie.empty:
            return movie.iloc[0]['movieId']
        else:
            return None

    movie_id_to_recommend = get_movie_id(selected_movie_title)

    movie_recommendations = []

    if movie_id_to_recommend is not None:
        similar_movies = movie_similarity_df[movie_id_to_recommend].sort_values(ascending=False)[1:11]
        movie_recommendations = similar_movies.index.tolist()

    st.sidebar.subheader('Top Movie Recommendations:')
    for i, movie_id in enumerate(movie_recommendations, 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            st.sidebar.write(f"{i}. Movie Title: {title}")

# User-Based Recommendation
elif recommendation_type == "User-Based":
    st.sidebar.header('User-Based Recommendation')
    user_id_to_recommend = st.sidebar.number_input('Enter User ID:', min_value=1, max_value=ratings_df['userId'].max(), value=1)

    def get_top_rated_movies(user_id, n):
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        top_rated_movies = user_ratings.nlargest(n, 'rating')

        # Join with movies_df to get movie titles
        top_rated_movies_with_titles = top_rated_movies.merge(movies_df, on='movieId', how='left')

        return top_rated_movies_with_titles

    top_rated_movies_df = get_top_rated_movies(user_id_to_recommend, 10)

    if not top_rated_movies_df.empty:
        st.sidebar.header(f'Top Rated Movies for User {user_id_to_recommend}:')
        for i, row in enumerate(top_rated_movies_df.iterrows(), 1):
            movie_title = row[1]['title']
            st.sidebar.write(f"{i}. Movie Title: {movie_title}")

    # Generate movie recommendations for the selected user
    recommendation_button = st.sidebar.button('Recommend Movies (User-Based)')
    
    if recommendation_button:
        user_movies = ratings_df[ratings_df['userId'] == user_id_to_recommend]['movieId'].unique()
        movies_to_recommend = [movie_id for movie_id in ratings_df['movieId'].unique() if movie_id not in user_movies]
        user_recommendations = []

        for movie_id in movies_to_recommend:
            estimated_rating = svd.predict(user_id_to_recommend, movie_id).est
            user_recommendations.append((movie_id, estimated_rating))

        user_recommendations.sort(key=lambda x: x[1], reverse=True)

        st.sidebar.subheader(f'Top Movie Recommendations for User {user_id_to_recommend}:')
        for i, (movie_id, estimated_rating) in enumerate(user_recommendations[:10], 1):
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                st.sidebar.write(f"{i}. Movie Title: {title}")

# Item-Based Recommendation
elif recommendation_type == "Item-Based":
    st.sidebar.header('Item-Based Recommendation')

    st.sidebar.subheader('Select a Movie:')
    selected_movie_title = st.sidebar.selectbox('', movies_df['title'])
    st.sidebar.subheader('')
    
    def get_movie_id(movie_title):
        movie = movies_df[movies_df['title'] == movie_title]
        if not movie.empty:
            return movie.iloc[0]['movieId']
        else:
            return None

    movie_id_to_recommend = get_movie_id(selected_movie_title)

    movie_recommendations = []

    if movie_id_to_recommend is not None:
        similar_movies = movie_similarity_df[movie_id_to_recommend].sort_values(ascending=False)[1:11]
        movie_recommendations = similar_movies.index.tolist()

    st.sidebar.subheader('Top Movie Recommendations:')
    for i, movie_id in enumerate(movie_recommendations, 1):
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            st.sidebar.write(f"{i}. Movie Title: {title}")

# Main content
st.sidebar.title('About')
st.sidebar.write("This Movie Recommender System helps you discover movies based on your preferences.")
st.sidebar.write("Data sources: [Add your data sources here]")
st.sidebar.write("Credits: [Add credits here]")

# Footer
st.sidebar.markdown(f"© {datetime.datetime.now().year} Your Company Name")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.write("This is a Streamlit app for movie recommendations.")


