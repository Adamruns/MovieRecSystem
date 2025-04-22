import pandas as pd
import torch
from torch import nn

# -------------------------------
# 1. Define the Collaborative Filtering Model (same as in training)
# -------------------------------
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

if __name__ == '__main__':
    # -------------------------------
    # 2. Load the Movies Data
    # -------------------------------
    movies_df = pd.read_csv('../data/raw/ml-latest/movies.csv')

    # -------------------------------
    # 3. Load the Training Checkpoint
    # -------------------------------
    checkpoint = torch.load(
        'collaborative_filtering_checkpoint.pt',
        map_location=torch.device('cpu'),
        weights_only=False
    )
    print("Checkpoint loaded.")

    user2idx = checkpoint['user2idx']
    movie2idx = checkpoint['movie2idx']
    # Create a reverse mapping for movie lookup.
    idx2movie = {idx: movie for movie, idx in movie2idx.items()}

    num_users = len(user2idx)
    num_movies = len(movie2idx)

    # -------------------------------
    # 4. Instantiate Model and Load Weights
    # -------------------------------
    embedding_size = 50
    model = CollaborativeFiltering(num_users, num_movies, embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Set device (CPU here; adjust if needed)
    device = torch.device('cpu')
    model.to(device)

    # -------------------------------
    # 5. Interactive Movie Recommendation Engine
    # -------------------------------
    print("\nMovie Recommendation Engine")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("Enter movies you like (comma separated): ")
        if user_input.lower().strip() == 'exit':
            break

        liked_titles = [title.strip() for title in user_input.split(',')]
        liked_movie_indices = []
        for title in liked_titles:
            # Perform case-insensitive substring search for matching titles.
            matches = movies_df[movies_df['title'].str.contains(title, case=False, na=False)]
            if matches.empty:
                print(f"No match found for movie: {title}")
            else:
                movie_id = matches.iloc[0]['movieId']
                if movie_id in movie2idx:
                    liked_movie_indices.append(movie2idx[movie_id])
                else:
                    print(f"Movie ID {movie_id} for '{title}' not found in our mapping.")

        if not liked_movie_indices:
            print("No valid movies found from your input. Please try again.\n")
            continue

        # -------------------------------
        # 6. Generate Recommendation
        # -------------------------------
        with torch.no_grad():
            movie_embeddings = model.movie_embedding.weight
            movie_bias = model.movie_bias.weight.squeeze()
            liked_embeddings = movie_embeddings[liked_movie_indices]
            pseudo_user_embedding = liked_embeddings.mean(dim=0)
            scores = torch.matmul(movie_embeddings, pseudo_user_embedding) + movie_bias

            # Exclude movies already liked.
            for idx in liked_movie_indices:
                scores[idx] = -float('inf')

            recommended_idx = torch.argmax(scores).item()

        recommended_movie_id = idx2movie[recommended_idx]
        recommended_row = movies_df[movies_df['movieId'] == recommended_movie_id]
        if not recommended_row.empty:
            recommended_title = recommended_row.iloc[0]['title']
            print(f"Recommended movie: {recommended_title}\n")
        else:
            print("Could not find the recommended movie in movies_df.\n")
