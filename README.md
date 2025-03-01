# Movie Recommendation System

## Overview
The Movie Recommendation System is a machine learning-based application that suggests movies to users based on their preferences. The system leverages collaborative filtering and content-based filtering techniques to provide personalized recommendations.

## Features
- Personalized movie recommendations
- Content-based filtering using TF-IDF
- Collaborative filtering using the K-Nearest Neighbors (KNN) algorithm
- Cosine similarity for finding similar movies
- Interactive user interface

## Tech Stack
- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, NLTK
- Dataset: MovieLens dataset

## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
Download the MovieLens dataset from [MovieLens](https://grouplens.org/datasets/movielens/) and place it in the `dataset/` directory.

## Usage
Run the following command to start the recommendation system:
```bash
python main.py
```

## Machine Learning Algorithm
The system uses:
1. **Content-Based Filtering**
   - Uses TF-IDF vectorization to analyze movie descriptions.
   - Computes cosine similarity to recommend similar movies.

2. **Collaborative Filtering (KNN Algorithm)**
   - Uses user-item interaction data to find similar users.
   - Recommends movies based on the preferences of similar users.

## Code Example
Below is an example of how the recommendation system works:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movie(movie_title, movies_df, tfidf_matrix):
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    similarities = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix)
    similar_movies = similarities.argsort()[0][-6:-1][::-1]
    return movies_df.iloc[similar_movies]['title'].values

movies_df = pd.read_csv("dataset/movies.csv")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['description'].fillna(''))

movie_recommendations = recommend_movie("Inception", movies_df, tfidf_matrix)
print("Recommended movies:", movie_recommendations)
```

## Contributors
- **Aman Saroj**

## License
This project is licensed under the MIT License.
