# MovieRecSystem

A lightweight web app that provides movie recommendations using a pre‑trained collaborative filtering model. Enter comma‑separated movie titles, and the app will suggest a new movie you might enjoy.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoint](#api-endpoint)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

MovieRecSystem is a minimal viable product (MVP) that integrates a PyTorch‑based collaborative filtering model into a Flask web application. Users enter comma‑separated movie titles in the front end, and the backend returns a recommendation. The UI also reformats titles with leading articles (e.g.,  
`"Christmas Story, A (1983)"` → `"A Christmas Story (1983)"`) and handles extra parentheses (e.g., `"Taste of Others, The (Le goût des autres) (2000)"`) for readability.

---

## Features

- **Simple Form‑based UI**  
  - Input box for comma‑separated movie titles  
  - Button to fetch recommendations  
  - Clean, centered layout with basic CSS styling  

- **Intelligent Title Formatting**  
  - Regex in the front end rearranges `<Title>, A/An/The (Year)` into `A/An/The <Title> (Year)`  
  - Supports extra parenthetical content (e.g., foreign titles)

- **Backend Recommendation**  
  - Loads a pre‑trained PyTorch collaborative filtering model  
  - Uses a movies CSV (`movies.csv`) for title lookup and ID mappings  
  - Returns JSON with either a recommendation or a specific “couldn’t find” error message  

---

## Technologies

- **Python 3.x**  
- **Flask** – lightweight web framework  
- **PyTorch** – model loading & inference  
- **Pandas** – CSV data loading & processing  
- **Axios** – AJAX requests from front end  
- **Gunicorn** – WSGI HTTP server for deployment  
- **Git & Git LFS** – version control & large‑file management  

---

## Project Structure

```
MovieRecSystem/
├── .gitignore
├── .gitattributes                 # LFS tracking
├── Procfile                       # web: gunicorn src.app:app --bind 0.0.0.0:$PORT
├── requirements.txt               # pip install requirements
├── Pipfile / Pipfile.lock         # Pipenv dependency management
├── data/
│   └── raw/ml-latest/
│       └── movies.csv             # Movie metadata
└── src/
    ├── app.py                     # Flask application & model integration
    ├── collaborative_filtering_checkpoint.pt
    ├── movie_recommender.py       # helper logic
    ├── preprocess_archive.py      # data preprocessing scripts
    ├── preprocess_interviews.py
    ├── preprocess_ml_latest.py
    ├── train_model.py             # model training script
    └── templates/
        └── index.html             # Front‑end HTML & JS
```

---

## Installation & Setup

1. Clone the repo and enter its folder:
   ```bash
   git clone https://github.com/YourUsername/MovieRecSystem.git
   cd MovieRecSystem
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If using Git LFS for the model checkpoint:
   ```bash
   git lfs install
   git lfs pull
   ```

---

## Usage

1. Start the Flask server:
   ```bash
   cd src
   python app.py
   ```
2. Open your browser at `http://127.0.0.1:5000`.
3. Enter comma‑separated movie titles and click **Get Recommendation**.
4. View results:
   - If matches are found:  
     `Recommended movie: The Shawshank Redemption (1994)`  
   - If no matches:  
     `Couldn't find a movie called "Unknown Movie"`

---

## API Endpoint

```
POST /recommend
Content-Type: application/x-www-form-urlencoded

movies=<comma-separated titles>
```

**Response** (JSON):

- Success:  
  ```json
  { "recommendation": "The Shawshank Redemption (1994)" }
  ```
- Not found:  
  ```json
  { "recommendation": "Couldn't find a movie called \"Unknown Movie\"" }
  ```

---

## Deployment

Deploy to any Python‑friendly host. Example using Render:

1. Push your repo to GitHub.
2. On Render.com, create a **New → Web Service**, connect your GitHub repo.
3. Build command:  
   ```bash
   pip install -r requirements.txt
   ```
4. Start command:  
   ```bash
   gunicorn src.app:app --bind 0.0.0.0:$PORT
   ```
5. Provision under the free tier and deploy.

---

## License

MIT © Adam Smith
