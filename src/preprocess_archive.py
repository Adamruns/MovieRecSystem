#!/usr/bin/env python
"""
Archive Data Analysis and Visualization Script

This script loads and analyzes the data in your archive directory. It expects at least:
  - a ratings.csv file (with columns: userId, movieId, rating, timestamp)
  - a keywords.csv file (with columns: movieId, keywords; where keywords is a comma-separated list)

The script performs the following:
  • Loads ratings data and converts Unix timestamps to datetime.
  • Computes basic statistics (number of users, movies, total ratings).
  • Visualizes the rating distribution.
  • Plots the trend in ratings over time.
  • Loads and processes keywords data (if available) and visualizes the top 20 keywords.
  • Computes and visualizes the relationship between the number of ratings and the average rating per movie.

Usage:
    python analyze_archive.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the path to your archive directory (modify if needed)
archive_dir = '../data/raw/archive/'
# Set the output directory for all generated files (change this as needed)
output_dir = '../data/processed/archive_outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------------
# Load Ratings Data
# --------------------------
ratings_file = os.path.join(archive_dir, 'ratings.csv')
if not os.path.exists(ratings_file):
    print(f"Ratings file not found in {archive_dir}")
    exit(1)

ratings = pd.read_csv(ratings_file)
print(f"Loaded ratings data with {len(ratings)} records.")

# Convert Unix timestamp to datetime
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Compute basic statistics
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Total ratings: {len(ratings)}")

# --------------------------
# Visualization: Rating Distribution
# --------------------------
plt.figure(figsize=(8, 6))
plt.hist(ratings['rating'], bins=np.arange(0.25, 5.5, 0.5), edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
plt.close()
print("Saved rating distribution plot as 'rating_distribution.png'.")

# --------------------------
# Visualization: Ratings Trend Over Time
# --------------------------
# Set datetime as index for resampling
ratings.set_index('datetime', inplace=True)
monthly_counts = ratings.resample('M').size()

plt.figure(figsize=(10, 6))
monthly_counts.plot()
plt.xlabel('Month')
plt.ylabel('Number of Ratings')
plt.title('Monthly Ratings Trend')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ratings_trend.png'))
plt.close()
print("Saved ratings trend plot as 'ratings_trend.png'.")

# Reset index for further analysis
ratings.reset_index(inplace=True)

# --------------------------
# Load and Analyze Keywords Data (if available)
# --------------------------
keywords_file = os.path.join(archive_dir, 'keywords.csv')
if os.path.exists(keywords_file):
    keywords = pd.read_csv(keywords_file)
    print(f"Loaded keywords data with {len(keywords)} records.")

    # Assume the keywords column is a comma-separated list for each movie.
    # Create a new column that contains the list of keywords.
    keywords['keyword_list'] = keywords['keywords'].apply(
        lambda x: [kw.strip() for kw in str(x).split(',')]
    )

    # Explode the list so each row corresponds to one keyword per movie
    keywords_exploded = keywords.explode('keyword_list')

    # Count the frequency of each keyword
    keyword_counts = keywords_exploded['keyword_list'].value_counts().head(20)

    plt.figure(figsize=(10, 6))
    keyword_counts.plot(kind='bar')
    plt.xlabel('Keyword')
    plt.ylabel('Frequency')
    plt.title('Top 20 Keywords')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keywords_frequency.png'))
    plt.close()
    print("Saved keyword frequency plot as 'keywords_frequency.png'.")
else:
    print(f"Keywords file not found in {archive_dir}.")

# --------------------------
# Additional Analysis: Average Rating vs. Number of Ratings per Movie
# --------------------------
movie_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(8, 6))
plt.scatter(movie_ratings['count'], movie_ratings['mean'], alpha=0.5)
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Number of Ratings per Movie')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'movie_ratings_scatter.png'))
plt.close()
print("Saved movie ratings scatter plot as 'movie_ratings_scatter.png'.")
