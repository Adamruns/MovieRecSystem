# MovieRecSystem

A lightweight web app that provides movie recommendations using a pre‑trained collaborative filtering model. Enter comma‑separated movie titles, and the app will suggest a new movie you might enjoy.


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
