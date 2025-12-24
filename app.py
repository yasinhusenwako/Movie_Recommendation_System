import streamlit as st
import pandas as pd
import requests
import pickle
import os
from typing import Optional
import logging
from functools import lru_cache


# Page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for k, v in (('watchlist', []), ('user_ratings', {}), ('last_recommendations', None), ('last_selected_movie', None)):
    st.session_state.setdefault(k, v)

# UI styling — hide any residual 'Advanced Filters' expander in the sidebar
st.markdown("""
<style>
[aria-label="Advanced Filters"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

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
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_score = list(enumerate(cosine_sim[idx]))
        sim_score = sorted(sim_score, reverse=True, key=lambda x: x[1])
        sim_score = sim_score[1:10]  # getting scores of 10 most similar movies
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


def add_to_watchlist(movie_data: pd.Series):
    movie_id = movie_data['movie_id']
    if movie_id not in st.session_state.watchlist:
        st.session_state.watchlist.append(movie_id)
        st.success(f"Added '{movie_data['title']}' to your watchlist!")
    else:
        st.info(f"'{movie_data['title']}' is already in your watchlist.")


def remove_from_watchlist(movie_id: int):
    if movie_id in st.session_state.watchlist:
        st.session_state.watchlist.remove(movie_id)
        st.success("Removed from watchlist!")


def get_watchlist_movies() -> pd.DataFrame:
    """Get watchlist movies as DataFrame."""
    if not st.session_state.watchlist:
        return pd.DataFrame()

    watchlist_movies = movies[movies['movie_id'].isin(
        st.session_state.watchlist)]
    return watchlist_movies


def display_movie_info(movie_data: pd.Series, width: int = 150, overview_len: int = 100):
    poster_url = fetch_poster(int(movie_data.get('movie_id', 0)))
    if poster_url:
        st.image(poster_url, width=width)
    else:
        st.image(
            f"https://via.placeholder.com/{width}x{int(width*1.5)}?text=No+Poster", width=width)

    st.markdown(f"{movie_data.get('title', '')}")
    if 'vote_average' in movie_data and pd.notna(movie_data['vote_average']):
        st.write(f" {round(float(movie_data['vote_average']), 1)}/10")
    if 'release_date' in movie_data and pd.notna(movie_data['release_date']):
        st.write(f"{str(movie_data['release_date'])[:4]}")

    overview = str(movie_data.get('overview', '') or '')
    if overview:
        st.caption(overview[:overview_len] +
                   ('...' if len(overview) > overview_len else ''))


col1, col2 = st.columns([3, 1])
with col1:
    st.title("🎬 Movie Recommendation System")
    st.write(
        "Content-based recommendations (TF‑IDF + cosine similarity) with posters from TMDB.")
with col2:
    st.write(f"**Movies:** {len(movies)}")
    st.write(f"**Watchlist:** {len(st.session_state.watchlist)}")

st.markdown('<div class="mrs-divider"></div>', unsafe_allow_html=True)

tab_reco, tab_watchlist, tab_about = st.tabs(
    ["Recommendations", "Watchlist", "About"])

# Sidebar
st.sidebar.title("🎛️ Controls")
st.sidebar.caption(
    "Use filters to narrow the catalog before selecting a movie.")
st.sidebar.markdown("---")

search_term = st.sidebar.text_input('🔍 Search for a movie:')

sort_mode = st.sidebar.selectbox(
    "Sort by",
    [
        "Relevance (Default)",
        "Rating (High to Low)",
        "Rating (Low to High)",
        "Newest first",
        "Oldest first",
        "Title (A → Z)",
    ],
)

# filtered list (used by recommendations tab)
filtered_movies = movies.copy()

if search_term:
    filtered_movies = filtered_movies[
        filtered_movies['title'].str.contains(
            search_term, case=False, na=False)
    ]

# Sorting
if sort_mode != "Relevance (Default)":
    if sort_mode == "Title (A → Z)":
        filtered_movies = filtered_movies.sort_values(
            by='title', ascending=True, na_position='last')
    elif sort_mode in ("Rating (High to Low)", "Rating (Low to High)") and 'vote_average' in filtered_movies.columns:
        asc = sort_mode == "Rating (Low to High)"
        filtered_movies = filtered_movies.assign(_rating=pd.to_numeric(
            filtered_movies['vote_average'], errors='coerce'))
        filtered_movies = filtered_movies.sort_values(
            by='_rating', ascending=asc, na_position='last').drop(columns=['_rating'])
    elif sort_mode in ("Newest first", "Oldest first") and 'release_date' in filtered_movies.columns:
        asc = sort_mode == "Oldest first"
        filtered_movies = filtered_movies.assign(_date=pd.to_datetime(
            filtered_movies['release_date'], errors='coerce'))
        filtered_movies = filtered_movies.sort_values(
            by='_date', ascending=asc, na_position='last').drop(columns=['_date'])

with tab_reco:
    left, right = st.columns([1.15, 2.0], gap="large")

    with left:
        st.markdown(" Pick a movie")
        if filtered_movies.empty:
            st.warning("No movies match your filters.")
            selected_movie = None
        else:
            dropdown_df = filtered_movies.head(500)
            selected_movie = st.selectbox(
                'Select a movie you like:',
                dropdown_df['title'].values,
                index=0
            )

        if selected_movie:
            selected_row = movies[movies['title'] == selected_movie].iloc[0]
            st.markdown(" Selected movie")
            st.markdown('<div class="mrs-card">', unsafe_allow_html=True)
            display_movie_info(selected_row)

            is_in_watchlist = int(
                selected_row['movie_id']) in st.session_state.watchlist
            btn_label = "Remove from watchlist" if is_in_watchlist else "Add to watchlist"
            if st.button(btn_label, key=f"wl_selected_{int(selected_row['movie_id'])}"):
                if is_in_watchlist:
                    remove_from_watchlist(int(selected_row['movie_id']))
                else:
                    add_to_watchlist(selected_row)

            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(" Recommendations")
        if selected_movie and st.button('🎬 Show Recommendations', type="primary"):
            with st.spinner('Finding recommendations for you...'):
                recommendations = get_recommendations(selected_movie)
                st.session_state.last_recommendations = recommendations
                st.session_state.last_selected_movie = selected_movie
        elif st.session_state.last_recommendations is not None and st.session_state.last_selected_movie == selected_movie:
            recommendations = st.session_state.last_recommendations
        else:
            recommendations = None

        if recommendations is not None:
            if not recommendations.empty:
                st.success(
                    f'Top 10 movie recommendations for "{st.session_state.last_selected_movie}":')

                for i in range(0, len(recommendations), 3):
                    cols = st.columns(3, gap="small")
                    for col, idx in zip(cols, range(i, min(i + 3, len(recommendations)))):
                        with col:
                            movie_data = recommendations.iloc[idx]
                            display_movie_info(movie_data)

                            movie_id = int(movie_data['movie_id'])
                            is_in_watchlist = movie_id in st.session_state.watchlist
                            small_label = "✅" if is_in_watchlist else "Add to watchlist"
                            if st.button(small_label, key=f"wl_reco_{movie_id}"):
                                if is_in_watchlist:
                                    remove_from_watchlist(movie_id)
                                else:
                                    add_to_watchlist(movie_data)
            else:
                st.error("No recommendations available.")
        else:
            st.info("Pick a movie on the left, then click Show Recommendations.")

with tab_watchlist:
    st.markdown("Your watchlist")
    wl = get_watchlist_movies()
    if wl.empty:
        st.info("Your watchlist is empty. Add movies from the Recommendations tab.")
    else:
        action_left, action_right = st.columns([1, 3])
        with action_left:
            if st.button("Clear watchlist", type="secondary"):
                st.session_state.watchlist = []
                st.rerun()
        with action_right:
            st.caption(
                "Tip: remove individual items below, or clear the whole list.")

        for i in range(0, len(wl), 3):
            cols = st.columns(3, gap="small")
            for col, idx in zip(cols, range(i, min(i + 3, len(wl)))):
                with col:
                    movie_data = wl.iloc[idx]
                    display_movie_info(movie_data)
                    movie_id = int(movie_data['movie_id'])
                    if st.button("Remove", key=f"wl_remove_{movie_id}"):
                        remove_from_watchlist(movie_id)
with tab_about:
    st.markdown("About")
    st.markdown(
        """
        - Model: Content-based filtering
        - Text features: genres + keywords + top cast + director (tags)
        - Vectorization: TF‑IDF
        - Similarity: cosine similarity
        - Posters: TMDB API
        """
    )
# Footer
st.markdown("---")
st.markdown(
    "*Powered by TMDB API | Content-based recommendations using TF-IDF and Cosine Similarity*")
