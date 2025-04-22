#!/usr/bin/env python
"""
Analysis Script for MovieLens ml-latest Dataset

This script loads the following files from an input directory:
    - ratings.csv
    - movies.csv
    - tags.csv
    - links.csv
    - genome-scores.csv
    - genome-tags.csv

It then computes basic statistics and creates several visualizations:
    • Distribution of ratings (ratings.csv)
    • Genre distribution (movies.csv; genres are pipe-separated)
    • Top 20 tags usage frequency (tags.csv)
    • Relevance score distribution (genome-scores.csv)
    • Average movie rating distribution for movies with a sufficient number of ratings

Usage:
    python analyze_ml_latest.py

You can change the input directory and output directory via the variables defined below.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the input directory containing the ml-latest files (change this as needed)
input_dir = '../data/raw/ml-latest/'

# Set the output directory for all generated plots (change this as needed)
output_dir = '../data/processed/ml_latest_outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define file paths by joining the input directory with file names
ratings_file = os.path.join(input_dir, 'ratings.csv')
movies_file = os.path.join(input_dir, 'movies.csv')
tags_file = os.path.join(input_dir, 'tags.csv')
links_file = os.path.join(input_dir, 'links.csv')
genome_scores_file = os.path.join(input_dir, 'genome-scores.csv')
genome_tags_file = os.path.join(input_dir, 'genome-tags.csv')

# Load CSV files
print("Loading data...")
ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)
tags = pd.read_csv(tags_file)
links = pd.read_csv(links_file)
genome_scores = pd.read_csv(genome_scores_file)
genome_tags = pd.read_csv(genome_tags_file)

print("Data loaded successfully:")
print(f" - Ratings: {ratings.shape}")
print(f" - Movies: {movies.shape}")
print(f" - Tags: {tags.shape}")
print(f" - Links: {links.shape}")
print(f" - Genome Scores: {genome_scores.shape}")
print(f" - Genome Tags: {genome_tags.shape}")

# Basic statistics
num_users = ratings['userId'].nunique()
num_movies_rated = ratings['movieId'].nunique()
print("\nBasic Statistics:")
print(f"Number of users: {num_users}")
print(f"Number of movies rated: {num_movies_rated}")
print(f"Total ratings: {len(ratings)}")

# Visualization 1: Rating Distribution
plt.figure(figsize=(8, 6))
ratings['rating'].hist(bins=np.arange(0.25, 5.5, 0.5), edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
plt.close()
print("Saved rating distribution plot as 'rating_distribution.png'.")

# Visualization 2: Genre Distribution
# The 'genres' column in movies.csv is pipe-separated
genres_series = movies['genres'].str.split('|').explode()
genre_counts = genres_series.value_counts()
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genre Distribution in Movies')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'genre_distribution.png'))
plt.close()
print("Saved genre distribution plot as 'genre_distribution.png'.")

# Visualization 3: Top 20 Tag Usage Frequency
tag_counts = tags['tag'].value_counts().head(20)
plt.figure(figsize=(12, 6))
tag_counts.plot(kind='bar')
plt.xlabel('Tag')
plt.ylabel('Frequency')
plt.title('Top 20 Tags Usage Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tag_usage.png'))
plt.close()
print("Saved tag usage frequency plot as 'tag_usage.png'.")

# Visualization 4: Genome Relevance Score Distribution
plt.figure(figsize=(8, 6))
plt.hist(genome_scores['relevance'], bins=50, edgecolor='black')
plt.xlabel('Relevance Score')
plt.ylabel('Frequency')
plt.title('Genome Relevance Score Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'genome_relevance_distribution.png'))
plt.close()
print("Saved genome relevance distribution plot as 'genome_relevance_distribution.png'.")

# Additional Analysis: Merge ratings with movies to compute average rating per movie
movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movie_ratings = movie_ratings.merge(movies, on='movieId', how='left')
# For reliable statistics, consider movies with at least 50 ratings
popular_movies = movie_ratings[movie_ratings['count'] >= 50]
plt.figure(figsize=(8, 6))
plt.hist(popular_movies['mean'], bins=20, edgecolor='black')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.title('Average Rating Distribution for Movies (>=50 ratings)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_rating_distribution.png'))
plt.close()
print("Saved average rating distribution plot as 'average_rating_distribution.png'.")

print("Analysis complete. All plots have been saved to the output directory.")
