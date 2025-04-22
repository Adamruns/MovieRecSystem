import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Define a PyTorch Dataset
# -------------------------------
class RatingsDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(dataframe['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# -------------------------------
# 2. Define the Collaborative Filtering Model
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
    # 3. Data Preprocessing and Setup
    # -------------------------------
    ratings_df = pd.read_csv('../data/raw/ml-latest/ratings.csv')
    movies_df = pd.read_csv('../data/raw/ml-latest/movies.csv')

    print("Ratings data preview:")
    print(ratings_df.head())
    print("\nMovies data preview:")
    print(movies_df.head())

    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    print(f"\nTotal unique users: {num_users}, Total unique movies: {num_movies}")

    # Create mappings from original IDs to consecutive indices.
    user2idx = {user: idx for idx, user in enumerate(user_ids)}
    movie2idx = {movie: idx for idx, movie in enumerate(movie_ids)}

    # Map the userId and movieId columns to new indices.
    ratings_df['user_idx'] = ratings_df['userId'].map(user2idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie2idx)

    # -------------------------------
    # 4. Create Dataset and DataLoader
    # -------------------------------
    dataset = RatingsDataset(ratings_df.sample(n=1000000, random_state=42))
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=1)

    # -------------------------------
    # 5. Instantiate the Model, Loss, and Optimizer
    # -------------------------------
    embedding_size = 50
    model = CollaborativeFiltering(num_users, num_movies, embedding_size)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Choose device: Here checking for Apple MPS support; adjust as needed.
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    # -------------------------------
    # 6. Training Loop
    # -------------------------------
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_users, batch_movies, batch_ratings in dataloader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_ratings = batch_ratings.to(device)

            optimizer.zero_grad()
            predictions = model(batch_users, batch_movies)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_users.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # -------------------------------
    # 7. Save the Training Checkpoint
    # -------------------------------
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'user2idx': user2idx,
        'movie2idx': movie2idx,
    }
    torch.save(checkpoint, 'collaborative_filtering_checkpoint.pt')
    print("Training checkpoint saved to 'collaborative_filtering_checkpoint.pt'")
