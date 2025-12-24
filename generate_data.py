import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Loading data files...")
try:
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    print("Data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure tmdb_5000_credits.csv and tmdb_5000_movies.csv are in the directory.")
    exit(1)

print("Merging datasets...")
movies = movies.merge(credits, left_on='title', right_on='title')

print("Selecting relevant columns...")
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


print("Processing genres...")
movies['genres'] = movies['genres'].apply(convert)

print("Processing keywords...")
movies['keywords'] = movies['keywords'].apply(convert)

print("Processing cast (top 3)...")
movies['cast'] = movies['cast'].apply(
    lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

print("Processing crew (director)...")
movies['crew'] = movies['crew'].apply(
    lambda x: [i['name'] if i['job'] == 'Director' else '' for i in ast.literal_eval(x)])

print("Creating tags...")
movies['tags'] = movies['genres'] + \
    movies['keywords']+movies['cast']+movies['crew']

print("Cleaning up data...")
movies = movies[['movie_id', 'title', 'overview', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

print("Creating TF-IDF matrix...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

print("Computing cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Saving data files...")
# Save movies dataframe
with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(movies.to_dict(), f)

# Save cosine similarity matrix
with open('cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Data files generated successfully!")
print(f"Movies shape: {movies.shape}")
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
print("Files created: movie_dict.pkl, cosine_sim.pkl")
