# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Streamlit, using TMDB dataset and machine learning techniques to provide personalized movie suggestions.

## 🌟 Features

- **Content-Based Recommendations**: Uses TF-IDF vectorization and cosine similarity to find similar movies
- **Interactive UI**: Modern Streamlit interface with movie posters and metadata
- **Search Functionality**: Search for specific movies in the database
- **Genre Filtering**: Filter movies by genre for better discovery
- **Rich Movie Cards**: Display ratings, release year, and overview
- **API Caching**: Efficient poster loading with LRU cache
- **Error Handling**: Robust error handling for API failures and missing data

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn (TF-IDF, Cosine Similarity)
- **Data Processing**: Pandas, NumPy
- **External API**: TMDB (The Movie Database)
- **Caching**: functools.lru_cache

## 📋 Prerequisites

- Python 3.8 or higher
- TMDB API key (get one from [TMDB API Settings](https://www.themoviedb.org/settings/api))

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Movie-Recommendation-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your TMDB API key
TMDB_API_KEY=your_actual_tmdb_api_key_here
```

### 4. Download Required Data Files

The TMDB dataset files are too large for GitHub. Download them from:

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
  - Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
  - Place both files in the project directory

### 5. Generate Processed Data Files

Run the data processing script to create the required pickle files:

```bash
python generate_data.py
```

This will create:

- `movie_dict.pkl` - Processed movie data with tags
- `cosine_sim.pkl` - Cosine similarity matrix for recommendations

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📊 Data Processing

The system processes movie data through the following steps:

1. **Data Loading**: Load TMDB movies and credits datasets
2. **Feature Engineering**: Combine genres, keywords, cast (top 3), and director
3. **Text Processing**: Create tags by joining all features
4. **Vectorization**: Apply TF-IDF vectorization to movie tags
5. **Similarity Calculation**: Compute cosine similarity matrix
6. **Storage**: Save processed data as pickle files

## 🎯 How It Works

1. **Movie Selection**: User selects a movie from the dropdown or searches for one
2. **Similarity Calculation**: System finds movies with similar content features
3. **Recommendation Display**: Shows top 10 most similar movies with posters and metadata
4. **Interactive Features**: Users can search and filter by genre

## 🔧 Configuration

### Environment Variables

- `TMDB_API_KEY`: Your TMDB API key for fetching movie posters

### File Structure

```
Movie-Recommendation-System/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── movie_dict.pkl        # Processed movie data
├── cosine_sim.pkl        # Cosine similarity matrix
├── tmdb_5000_movies.csv  # Raw movie data
├── tmdb_5000_credits.csv # Raw credits data
└── Movie_Recommendation_System.ipynb # Data processing notebook
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**

   - Ensure `movie_dict.pkl` and `cosine_sim.pkl` exist
   - Run the notebook to regenerate them if needed

2. **API Key Issues**

   - Verify your TMDB API key is valid
   - Check that it's properly set in environment variables

3. **Poster Loading Errors**

   - Some movies may not have posters available
   - The system gracefully handles missing posters

4. **Memory Issues**
   - The application uses caching to optimize performance
   - Large datasets may require more RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Enhancements to Consider

- [ ] User rating system
- [ ] Collaborative filtering
- [ ] Movie popularity trends
- [ ] User preference learning
- [ ] Advanced filtering options
- [ ] Export recommendations
- [ ] Mobile-responsive design

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TMDB for providing the movie database API
- Streamlit team for the excellent web app framework
- Scikit-learn for machine learning tools

---

_Built with ❤️ for movie enthusiasts everywhere!_
