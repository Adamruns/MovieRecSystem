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

temp_df = pd.read_csv(
    DATA_PATH,
    usecols=['movieId', 'title', 'genres'],
    dtype={'movieId': int, 'title': 'string', 'genres': 'string'}
)
meta = {
    row.movieId: {'title': row.title, 'genres': row.genres}
    for row in temp_df.itertuples(index=False)
}
# Create a list for substring search
movies_list = [{'movieId': mid, 'title': data['title']} for mid, data in meta.items()]
del temp_df  # drop heavy DataFrame

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

def get_recommendations(user_input, n=5):
    # 1️⃣ parse liked titles using lightweight list and meta
    liked_titles = [t.strip() for t in user_input.split(',')]
    liked_idxs, not_found = [], []
    for title in liked_titles:
        # substring search over lightweight list
        matches = [m for m in movies_list if title.lower() in m['title'].lower()]
        if not matches:
            not_found.append(title)
        else:
            mid = matches[0]['movieId']
            if mid in movie2idx:
                liked_idxs.append(movie2idx[mid])

    if not liked_idxs:
        # return a string here if you prefer error handling as before
        return f"Couldn't find: {', '.join(not_found)}"

    # 2️⃣ compute scores just like before
    with torch.no_grad():
        emb = model.movie_embedding.weight       # (#movies × dim)
        bias = model.movie_bias.weight.squeeze() # (#movies,)
        liked_embs = emb[liked_idxs]
        pseudo_user = liked_embs.mean(dim=0)     # (dim,)

        scores = emb @ pseudo_user + bias
        # exclude already-liked
        for idx in liked_idxs:
            scores[idx] = -float('inf')

        # 3️⃣ grab top-N instead of argmax
        top_scores, top_idxs = torch.topk(scores, k=n)

    # 4️⃣ map back to movie IDs & then to titles using meta
    top_movie_ids = [ idx2movie[i.item()] for i in top_idxs ]
    # Map top IDs to titles using lightweight metadata
    recommended = [meta[mid]['title'] for mid in top_movie_ids]

    return recommended


# -------------------------------
# 3. Define Flask Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_input = request.form.get('movies', '')
    # pull `n` from the form (default to 5):
    n = request.form.get('n', 5, type=int)

    result = get_recommendations(movie_input, n=n)

    # if result is still a string (error), you can detect it:
    if isinstance(result, str):
        return jsonify({'error': result}), 400

    return jsonify({'recommendations': result})


if __name__ == '__main__':
    app.run(debug=True)
