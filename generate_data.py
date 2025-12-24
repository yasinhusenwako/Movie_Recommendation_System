import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

try:
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
except FileNotFoundError as e:
    print(f"Required CSVs not found: {e}")
    exit(1)

print("Merging datasets...")
movies = movies.merge(credits, left_on='title', right_on='title')

print("Selecting relevant columns...")
movies = movies[['movie_id', 'title', 'overview', 'release_date', 'vote_average',
                 'genres', 'keywords', 'cast', 'crew']]


def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]


print("Processing genres...")
movies['genres'] = movies['genres'].apply(convert)

print("Processing keywords...")
movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(
    lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

movies['crew'] = movies['crew'].apply(
    lambda x: [i['name'] for i in ast.literal_eval(x) if i.get('job') == 'Director'])

movies['tags'] = movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['crew']
movies = movies[['movie_id', 'title', 'overview',
                 'release_date', 'vote_average', 'genres', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(movies.to_dict(), f)
with open('cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Generated: movie_dict.pkl, cosine_sim.pkl")
