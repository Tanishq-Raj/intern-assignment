# Netflix Catalog Q&A System

A production-grade backend and natural-language Q&A system built on FastAPI, SQLite, and an offline-capable Retrieval-Augmented Generation (RAG) pipeline for the Netflix catalog.

---

## Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `git` (optional)

### Installation
1. **Clone or navigate to the repository directory**:
   ```bash
   cd /path/to/intern-assignment-main
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration (optional)**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *(Note: The system works completely out-of-the-box in offline mode without setting any external API keys.)*

---

## How to Run

### 1. Ingest Dataset into SQLite
Run the ingestion script to clean the CSV, construct relational tables and indexes, and output the data-quality audit report:
```bash
python ingest.py
```
This generates `data/netflix.db` containing `titles`, `title_countries`, `title_genres`, `title_cast`, and `title_directors`.

### 2. Pre-build RAG Index (Optional)
Pre-compute and cache the dense semantic vector embeddings using local `sentence-transformers` (`all-MiniLM-L6-v2`):
```bash
python -m rag.indexer
```
*(If this step is skipped, the application will automatically build and cache the index upon the first query.)*

### 3. Start the Backend API
Start the FastAPI application using Uvicorn:
```bash
uvicorn api.main:app --reload --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI / Swagger UI: `http://127.0.0.1:8000/docs`

---

## API Endpoints & Examples

### 1. `GET /titles`
List titles with query parameter filtering and deterministic pagination.
- **Query parameters**:
  - `country` (str, optional): e.g. `India` (matches both single and co-productions)
  - `release_year` (int, optional): e.g. `2018`
  - `type` (str, optional): `Movie` or `TV Show`
  - `rating` (str, optional): e.g. `TV-MA`, `PG-13`
  - `page` (int, default `1`, ge=1)
  - `page_size` (int, default `20`, ge=1, le=100)
- **Example request**:
  ```bash
  curl -X GET "http://127.0.0.1:8000/titles?country=India&release_year=2018&type=Movie&page=1&page_size=2"
  ```
- **Example response**:
  ```json
  {
    "total": 85,
    "page": 1,
    "page_size": 2,
    "total_pages": 43,
    "titles": [
      {
        "show_id": 80180851,
        "type": "Movie",
        "title": "Gemini",
        "director": "P.K. Baaburaaj",
        "cast": "Esther Anil, Sijoy Varghese...",
        "country": "India",
        "date_added": "July 1, 2018",
        "date_added_iso": "2018-07-01",
        "release_year": 2018,
        "rating": "TV-PG",
        "duration": "130 min",
        "listed_in": "Dramas",
        "description": "Despite growing up with traits that set her apart..."
      }
    ]
  }
  ```

### 2. `GET /titles/{show_id}`
Fetch a single title by its unique `show_id`. Returns `404 Not Found` if the identifier is absent.
- **Example request**:
  ```bash
  curl -X GET "http://127.0.0.1:8000/titles/81075235"
  ```
- **Example response**:
  ```json
  {
    "show_id": 81075235,
    "type": "Movie",
    "title": "Sarkar",
    "director": "A.R. Murugadoss",
    "cast": "Vijay, Varalakshmi Sarathkumar, Keerthi Suresh...",
    "country": "India",
    "date_added": "March 2, 2019",
    "date_added_iso": "2019-03-02",
    "release_year": 2018,
    "rating": "TV-MA",
    "duration": "162 min",
    "listed_in": "Action & Adventure, Dramas, International Movies",
    "description": "A ruthless businessman's mission to expose electoral fraud..."
  }
  ```

### 3. `GET /stats`
Return catalog summary statistics: total titles, breakdown by type, and top 10 countries.
- **Example request**:
  ```bash
  curl -X GET "http://127.0.0.1:8000/stats"
  ```
- **Example response**:
  ```json
  {
    "total_titles": 6234,
    "count_by_type": {
      "Movie": 4265,
      "TV Show": 1969
    },
    "count_by_country": [
      {"country": "United States", "count": 2610},
      {"country": "India", "count": 838},
      {"country": "United Kingdom", "count": 602},
      {"country": "Canada", "count": 318},
      {"country": "France", "count": 271},
      {"country": "Japan", "count": 231},
      {"country": "Spain", "count": 178},
      {"country": "South Korea", "count": 162},
      {"country": "Germany", "count": 151},
      {"country": "Mexico", "count": 129}
    ]
  }
  ```

### 4. `POST /ask`
Natural language question grounded strictly in the catalogue. Returns an answer accompanied by cited `show_ids` and source metadata.
- **Example request**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/ask" \
       -H "Content-Type: application/json" \
       -d '{"question": "Suggest me a comedy movie from India with a strong female lead"}'
  ```
- **Example response**:
  ```json
  {
    "question": "Suggest me a comedy movie from India with a strong female lead",
    "answer": "Based on the Netflix catalogue, here are relevant titles for your query...\n• Queens of Comedy (show_id: 80229190, TV Show, 2017): Eight women perform in front of celebrity judges...",
    "show_ids": [80229190, 80219143, 80203791],
    "sources": [
      {
        "show_id": 80229190,
        "title": "Queens of Comedy",
        "type": "TV Show",
        "release_year": 2017,
        "country": null,
        "rating": "TV-MA",
        "score": 0.1667
      }
    ]
  }
  ```

---

## How to Test

Run the automated test suite covering data ingestion, idempotency, API filtering, pagination, error handling, RAG retrieval, and mocked LLM generation:
```bash
python run_tests.py
```
*(Or directly run `pytest -v`)*

---

## Known Limitations

1. **TF-IDF Semantic Granularity**:
   While TF-IDF with n-grams is fast and operates offline, it relies on lexical overlap and keyword matching. Complex conceptual queries with no matching vocabulary (e.g., *"movies about existential dread"*) perform better with dense neural embeddings.
2. **Synchronous SQLite File Locking**:
   SQLite is lightweight and zero-configuration, but does not support concurrent write operations across multiple processes. Under heavy concurrent write loads, connection pooling with PostgreSQL would be necessary.
3. **No Automatic Query Intent Separation**:
   Natural language queries containing compound boolean conditions (e.g., *"movies after 2015 from France"*) are processed via vector similarity rather than decomposing into SQL `WHERE release_year > 2015 AND country = 'France'`. Implementing structured function-calling or filter extraction is the recommended next step.
4. **Duplicate Title Identification**:
   *Sarkar* appears twice in the dataset under two distinct `show_id` values (81075235 and 81072516). The system retains both to protect upstream catalog ID integrity, but a deduplication pipeline would be needed if deduplicated presentation is desired.
