# app.py
import os

from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
from torch import nn

app = Flask(__name__)

# -------------------------------
# 1. Load Model and Movies Data
# -------------------------------


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH    = os.path.join(PROJECT_ROOT,
                            'data', 'raw', 'ml-latest', 'movies.csv')

movies_df = pd.read_csv(DATA_PATH)

base_dir       = os.path.dirname(os.path.abspath(__file__))
checkpoint_fp = os.path.join(base_dir, 'collaborative_filtering_checkpoint.pt')

checkpoint = torch.load(checkpoint_fp,
                        map_location=torch.device('cpu'),
                        weights_only=False)
print("Checkpoint loaded.")

# Create mappings from the checkpoint
user2idx = checkpoint['user2idx']
movie2idx = checkpoint['movie2idx']
idx2movie = {idx: movie for movie, idx in movie2idx.items()}

num_users = len(user2idx)
num_movies = len(movie2idx)
embedding_size = 50  # Adjust if you used a different embedding size


# Define the Collaborative Filtering Model (same as during training)
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(CollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.movie_embedding.weight, 0, 0.1)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.movie_bias.weight, 0)

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        interaction = (user_emb * movie_emb).sum(1, keepdim=True)
        user_b = self.user_bias(user)
        movie_b = self.movie_bias(movie)
        rating = interaction + user_b + movie_b
        return rating.squeeze()


# Instantiate and load model weights
model = CollaborativeFiltering(num_users, num_movies, embedding_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(torch.device('cpu'))


# -------------------------------
# 2. Define Recommendation Logic
# -------------------------------

def get_recommendation(user_input):
    liked_titles = [title.strip() for title in user_input.split(',')]
    liked_movie_indices = []
    not_found = []

    for title in liked_titles:
        # Perform case-insensitive substring search for matching titles.
        matches = movies_df[movies_df['title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            not_found.append(title)
        else:
            movie_id = matches.iloc[0]['movieId']
            if movie_id in movie2idx:
                liked_movie_indices.append(movie2idx[movie_id])

    # If no valid movies were matched, return a specific error message.
    if not liked_movie_indices:
        if len(not_found) == 1:
            return f"Couldn't find a movie called \"{not_found[0]}\""
        elif len(not_found) > 1:
            return f"Couldn't find movies called \"{', '.join(not_found)}\""
        else:
            return "No valid movies found."

    with torch.no_grad():
        movie_embeddings = model.movie_embedding.weight
        movie_bias = model.movie_bias.weight.squeeze()
        liked_embeddings = movie_embeddings[liked_movie_indices]
        pseudo_user_embedding = liked_embeddings.mean(dim=0)
        scores = torch.matmul(movie_embeddings, pseudo_user_embedding) + movie_bias

        # Exclude movies already liked by setting their scores to negative infinity.
        for idx in liked_movie_indices:
            scores[idx] = -float('inf')

        recommended_idx = torch.argmax(scores).item()

    recommended_movie_id = idx2movie[recommended_idx]
    rec_row = movies_df[movies_df['movieId'] == recommended_movie_id]
    if not rec_row.empty:
        return rec_row.iloc[0]['title']
    return "Recommendation not found."


# -------------------------------
# 3. Define Flask Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_input = request.form.get('movies', '')
    result = get_recommendation(movie_input)
    return jsonify({'recommendation': result})


if __name__ == '__main__':
    app.run(debug=True)
