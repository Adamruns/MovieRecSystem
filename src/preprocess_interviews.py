#!/usr/bin/env python
"""
Data Collection and Preprocessing Script for Movie Recommendation System

This script loads raw interview data (e.g., from interviews.csv), cleans and preprocesses the data,
performs basic feature engineering and visualization, and saves a processed dataset that can be fed into PyTorch.

Steps performed:
• Load the CSV file containing interview data
• Check and handle missing values
• Convert categorical fields (e.g., gender, streaming_service) into one-hot encoded features
• Process multi-valued columns (preferred_genres, recommendation_source, decision_factors, common_dislikes)
  by splitting on commas and applying multi-label binarization
• Map textual watch frequency to a numeric scale
• Generate visualizations to help identify trends and validate data quality
• Save the cleaned/processed data to a CSV file

Usage:
    python preprocess_data.py

Note: Adjust file paths as necessary.
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# Load raw interview data
# Set the output directory for all generated files (change this as needed)
output_dir = '../data/processed/interviews_outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


try:
    df = pd.read_csv('../data/raw/interviews.csv')
    print('Data loaded successfully from interviews.csv')
except Exception as e:
    print(f'Error loading data: {e}')
    exit(1)

# Display initial info
print('Initial data preview:')
print(df.head())

# Check for missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# Map watch_frequency to a numeric scale
# Define a mapping: 'Weekly' -> 1, 'Twice weekly' -> 2, 'Monthly' -> 0.5
def map_frequency(freq):
    if isinstance(freq, str):
        freq_lower = freq.lower().strip()
        mapping = {
            'weekly': 1,
            'twice weekly': 2,
            'monthly': 0.5
        }
        return mapping.get(freq_lower, np.nan)
    return np.nan

# Create a new numeric column for watch frequency
if 'watch_frequency' in df.columns:
    df['watch_frequency_numeric'] = df['watch_frequency'].apply(map_frequency)
else:
    print('Column watch_frequency not found in data.')

# Function to split comma-separated text and strip whitespace
def split_and_strip(text):
    if isinstance(text, str):
        return [x.strip() for x in text.split(',')]
    return []

# Process multi-valued categorical columns using MultiLabelBinarizer
# 1. preferred_genres
if 'preferred_genres' in df.columns:
    df['preferred_genres_list'] = df['preferred_genres'].apply(split_and_strip)
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb_genres.fit_transform(df['preferred_genres_list']),
                                  columns=['genre_' + g for g in mlb_genres.classes_])
    df = pd.concat([df, genres_encoded], axis=1)
else:
    print('Column preferred_genres not found in data.')

# 2. recommendation_source
if 'recommendation_source' in df.columns:
    df['recommendation_source_list'] = df['recommendation_source'].apply(split_and_strip)
    mlb_rec = MultiLabelBinarizer()
    rec_encoded = pd.DataFrame(mlb_rec.fit_transform(df['recommendation_source_list']),
                               columns=['rec_src_' + s for s in mlb_rec.classes_])
    df = pd.concat([df, rec_encoded], axis=1)
else:
    print('Column recommendation_source not found in data.')

# 3. decision_factors
if 'decision_factors' in df.columns:
    df['decision_factors_list'] = df['decision_factors'].apply(split_and_strip)
    mlb_decision = MultiLabelBinarizer()
    decision_encoded = pd.DataFrame(mlb_decision.fit_transform(df['decision_factors_list']),
                                    columns=['dec_factor_' + s for s in mlb_decision.classes_])
    df = pd.concat([df, decision_encoded], axis=1)
else:
    print('Column decision_factors not found in data.')

# 4. common_dislikes
if 'common_dislikes' in df.columns:
    df['common_dislikes_list'] = df['common_dislikes'].apply(split_and_strip)
    mlb_dislikes = MultiLabelBinarizer()
    dislikes_encoded = pd.DataFrame(mlb_dislikes.fit_transform(df['common_dislikes_list']),
                                    columns=['dislike_' + s for s in mlb_dislikes.classes_])
    df = pd.concat([df, dislikes_encoded], axis=1)
else:
    print('Column common_dislikes not found in data.')

# For other categorical columns, use one-hot encoding
# Process gender
if 'gender' in df.columns:
    df = pd.concat([df, pd.get_dummies(df['gender'], prefix='gender')], axis=1)
else:
    print('Column gender not found in data.')

# Process streaming_service
if 'streaming_service' in df.columns:
    df = pd.concat([df, pd.get_dummies(df['streaming_service'], prefix='streaming')], axis=1)
else:
    print('Column streaming_service not found in data.')

# Drop intermediate columns that are no longer needed
cols_to_drop = ['preferred_genres', 'preferred_genres_list', 'recommendation_source', 'recommendation_source_list',
                'decision_factors', 'decision_factors_list', 'common_dislikes', 'common_dislikes_list', 'watch_frequency']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

df_processed = df.drop(columns=existing_cols_to_drop)

# Display processed data preview
print('\nProcessed data preview:')
print(df_processed.head())

# Data Visualization
# 1. Gender distribution
plt.figure(figsize=(6,4))
if 'gender' in df.columns:
    df['gender'].value_counts().plot(kind='bar', title='Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    plt.close()
    print('Saved gender distribution plot as gender_distribution.png')

# 2. Preferred genres count
if 'preferred_genres_list' in df.columns or any(col.startswith('genre_') for col in df_processed.columns):
    # Sum up one-hot encoded genre columns
    genre_columns = [col for col in df_processed.columns if col.startswith('genre_')]
    if genre_columns:
        genre_counts = df_processed[genre_columns].sum().sort_values(ascending=False)
        plt.figure(figsize=(8,4))
        genre_counts.plot(kind='bar', title='Preferred Genres Count')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'preferred_genres.png'))
        plt.close()
        print('Saved preferred genres plot as preferred_genres.png')

# 3. Watch frequency distribution
if 'watch_frequency_numeric' in df_processed.columns:
    plt.figure(figsize=(6,4))
    df_processed['watch_frequency_numeric'].plot(kind='hist', bins=5, title='Watch Frequency Distribution')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'watch_frequency_distribution.png'))
    plt.close()
    print('Saved watch frequency distribution plot as watch_frequency_distribution.png')

# Save the processed dataset for later use (e.g., feeding into PyTorch)
processed_file = os.path.join(output_dir, 'processed_data.csv')
df_processed.to_csv(processed_file, index=False)
print(f'\nProcessed data saved to {processed_file}')
