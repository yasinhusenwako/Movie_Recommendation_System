import streamlit as st
import pandas as pd
import requests
import pickle
import os
from typing import Optional, List
import logging
from functools import lru_cache
import time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
try:
    with open('movie_dict.pkl', 'rb') as f:
        movies_dict = pickle.load(f)
    # Convert dict back to DataFrame
    movies = pd.DataFrame(movies_dict)
    # Load cosine similarity matrix
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Required data files not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


def get_recommendations(title: str, cosine_sim=cosine_sim) -> pd.Series:
    """Get movie recommendations based on title."""
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_score = list(enumerate(cosine_sim[idx]))
        sim_score = sorted(sim_score, reverse=True, key=lambda x: x[1])
        sim_score = sim_score[1:11]  # getting scores of 10 most similar movies
        movie_indices = [i[0] for i in sim_score]
        return movies.iloc[movie_indices]
    except IndexError:
        st.error(f"Movie '{title}' not found in database.")
        return pd.Series()
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        st.error("An error occurred while getting recommendations.")
        return pd.Series()


@lru_cache(maxsize=1000)
def fetch_poster(movie_id: int) -> Optional[str]:
    """Fetch movie poster from TMDB API with caching."""
    api_key = os.getenv('TMDB_API_KEY', "54083c6f5142c6c6555c6740dc776e28")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path
        else:
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for movie {movie_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching poster for movie {movie_id}: {e}")
        return None


def display_movie_info(movie_data: pd.Series):
    """Display movie information in a card format."""
    poster_url = fetch_poster(movie_data['movie_id'])

    if poster_url:
        st.image(poster_url, width=150, use_column_width=False)
    else:
        st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)

    st.markdown(f"**{movie_data['title']}**")

    # Display additional metadata if available
    if 'vote_average' in movie_data and pd.notna(movie_data['vote_average']):
        rating = round(movie_data['vote_average'], 1)
        st.write(f"⭐ {rating}/10")

    if 'release_date' in movie_data and pd.notna(movie_data['release_date']):
        year = movie_data['release_date'][:4]
        st.write(f"📅 {year}")

    st.caption(movie_data['overview'][:100] + "..." if len(
        str(movie_data['overview'])) > 100 else movie_data['overview'])


# Main UI
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title('🎬 Movie Recommendation System')

# Sidebar for additional features
st.sidebar.title('🎯 Features')

# Search functionality
search_term = st.sidebar.text_input('🔍 Search for a movie:')

# Filter options
if 'genres' in movies.columns:
    all_genres = set()
    for genres_list in movies['genres'].dropna():
        if isinstance(genres_list, list):
            all_genres.update(genres_list)
    selected_genre = st.sidebar.selectbox(
        '🎭 Filter by Genre:', ['All'] + sorted(list(all_genres)))
else:
    selected_genre = 'All'

# Filter movies based on search and genre
filtered_movies = movies.copy()

if search_term:
    filtered_movies = filtered_movies[
        filtered_movies['title'].str.contains(
            search_term, case=False, na=False)
    ]

if selected_genre != 'All' and 'genres' in movies.columns:
    filtered_movies = filtered_movies[
        filtered_movies['genres'].apply(
            lambda x: selected_genre in x if isinstance(x, list) else False)
    ]

# Movie selection
if not filtered_movies.empty:
    selected_movie = st.selectbox(
        'Select a movie you like:',
        filtered_movies['title'].values,
        index=0
    )
else:
    st.warning("No movies found matching your criteria.")
    selected_movie = None

if selected_movie and st.button('🎬 Show Recommendations', type="primary"):
    with st.spinner('Finding recommendations for you...'):
        recommendations = get_recommendations(selected_movie)

        if not recommendations.empty:
            st.success(f'Top 10 movie recommendations for "{selected_movie}":')

            # Display recommendations in a grid
            for i in range(0, len(recommendations), 5):
                cols = st.columns(5)

                for j, (col, idx) in enumerate(zip(cols, range(i, min(i + 5, len(recommendations))))):
                    with col:
                        movie_data = recommendations.iloc[idx]
                        display_movie_info(movie_data)
        else:
            st.error("No recommendations available.")

# Footer
st.markdown("---")
st.markdown(
    "*Powered by TMDB API | Content-based recommendations using TF-IDF and Cosine Similarity*")
