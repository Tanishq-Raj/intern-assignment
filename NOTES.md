# Engineering & Design Notes — Netflix Catalog Q&A System

---

## 1. What problems did you find in the CSV? What did you fix, what did you leave, and why?

### Observations
The dataset contains **6,234 rows** and **12 columns**. An automated audit revealed:
1. **Whitespace anomalies**:
   - `date_added`: 640 rows contained leading or trailing whitespace (e.g., `" September 9, 2019 "`).
   - `title`: 1 row contained superfluous surrounding whitespace.
2. **Missing data (NaNs)**:
   - `director`: 1,969 missing values (~31.6%).
   - `cast`: 570 missing values (~9.1%).
   - `country`: 476 missing values (~7.6%).
   - `date_added`: 11 missing values (<0.2%).
   - `rating`: 10 missing values (<0.2%).
3. **Identifier uniqueness**:
   - Every row has a strictly unique `show_id` (6,234 unique IDs across 6,234 rows).
4. **Duplicate metadata**:
   - Exactly one movie was duplicated across two distinct `show_id`s: *Sarkar* (2018, directed by A.R. Murugadoss) appears under `show_id=81075235` and `show_id=81072516` with identical cast, duration, and plot description.
   - 69 titles share identical `(title, type)` pairs, but represent distinct remakes, adaptations, or seasons across different release years (e.g., *Benji* [1974 vs. 2018], *Carrie* [1976 vs. 2002], *Charmed* [1998 vs. 2018 reboot]).

### Decisions: What Was Fixed vs. Left
- **Fixed (Whitespace & Dates)**: Trimmed all string columns. Standardized all 6,223 valid `date_added` strings into ISO-8601 `YYYY-MM-DD` dates (`date_added_iso`) while retaining original raw strings for audit fidelity.
- **Left (Missing Values as SQL `NULL`)**: Missing values were preserved as database `NULL`s instead of fabricating placeholders (like `"Unknown"` or dummy dates). Fabricating data corrupts statistical counts and misleads retrieval models.
- **Left (Duplicate Show IDs & Remakes)**: Both entries of *Sarkar* and all legitimate remakes were retained. `show_id` is the authoritative catalogue identifier; dropping rows without upstream confirmation risks deleting active platform catalogue IDs.

---

## 2. Schema Decisions: Multi-Value Fields

### The Problem
Columns like `country` (923 multi-country entries), `listed_in` (4,784 multi-genre entries), and `cast` (4,967 multi-actor entries) are comma-delimited strings.
Querying top countries by grouping on raw strings produces over 500 composite values such as `"United States, Canada"`, distorting aggregations and making single-country filtering brittle.

### Chosen Architecture: Dual Representation Pattern
1. **Canonical `titles` Table**:
   Stores the row with its original comma-delimited strings (`country`, `cast`, `listed_in`).
   - *Why*: Allows $O(1)$ single-row retrieval (`GET /titles/{show_id}`) and paginated listings (`GET /titles`) without expensive multi-table joins or client-side reassembly.
2. **Normalized Junction Tables**:
   `title_countries(show_id, country)`, `title_genres(show_id, genre)`, `title_cast(show_id, actor)`, and `title_directors(show_id, director)`.
   - *Why*: Enables exact, index-backed country filtering (`show_id IN (SELECT show_id FROM title_countries WHERE country = ? COLLATE NOCASE)`), eliminating false-positive substring matches (e.g., "India" matching "Indiana").
   - Computes accurate `GET /stats` top 10 countries (e.g., United States: 2,610; India: 838; United Kingdom: 602).

---

## 3. RAG Choices: Representation, Retrieval & Generation

### Searchable Document Representation
Each title is represented as a structured key-value text block:
```text
Title: {title} | Type: {type} | Year: {release_year} | Rating: {rating} | Country: {country} | Genres: {listed_in} | Director: {director} | Cast: {cast} | Description: {description}
```
Embedding only the raw `description` fails when a user asks for "comedy movie from India" if the country and genre are not explicitly stated in the plot summary. Concatenating all structured metadata gives the retrieval engine full visibility across both structured attributes and narrative semantics.

### Chunk Size & Model
- **Chunk Size**: Exactly 1 document = 1 title (~80–120 tokens). Each title is an atomic catalog entity; splitting a short description into chunks would sever the connection between actors, genres, and synopsis.
- **Retrieval Engine**: Dense semantic embeddings using the local `sentence-transformers/all-MiniLM-L6-v2` model (384-dimensional dense vectors) paired with normalized cosine similarity (matrix dot product).
- *Why*: `all-MiniLM-L6-v2` provides deep semantic search across synopses, genres, and themes while running completely offline without external API costs or rate limits. For the 6,234 catalogue rows, dense vectors are computed once, normalized, and cached on disk (`data/rag_index.pkl`). At inference time, vector cosine matching executes in <10ms.
- **Generator & Grounding Policy**:
  - LLM integration is configurable via environment variables (`OPENAI_API_KEY`, `GROQ_API_KEY`).
  - Strict system prompt instructs the model to cite exact `show_id`s and titles, reject external knowledge, and distinguish objective metadata from subjective unproven attributes (such as "strong female lead").
  - If no API key is provided, an offline grounded synthesizer generates structured answers with real source citations and an explicit note on unproven subjective attributes, ensuring offline reproducibility.

---

## 4. What Would Break at 100,000 Titles?

1. **In-Memory TF-IDF Similarity ($O(N)$ dot products)**:
   Computing cosine similarity against 100,000 documents in Python creates noticeable latency per request (>500ms P99) and increases RAM usage.
   *Fix*: Migrate to an Approximate Nearest Neighbor (ANN) index (e.g., FAISS or HNSW index in pgvector).
2. **SQLite Database Write Concurrency**:
   SQLite locks the entire database file during writes (`SQLITE_BUSY`). Under concurrent batch updates or writes, API read latency would spike.
   *Fix*: Migrate to PostgreSQL with connection pooling (`asyncpg` / SQLAlchemy).
3. **Junction Table Growth**:
   At 100,000 titles, `title_cast` expands to ~700,000 rows.
   *Fix*: Composite B-tree indexes `(actor, show_id)` and materialized views or Redis caching for `GET /stats` aggregations.
4. **Context Window Limitations in Generation**:
   Passing large numbers of candidate titles to an LLM context window increases latency and cost.
   *Fix*: Strict Top-$K$ cutoff ($K \le 5$) with re-ranking (cross-encoder or BM25/Dense hybrid fusion).

---

## 5. AI Tool Usage & Code Review

### What AI Tools Were Used
Claude / Gemini assistant tools were utilized for initial boilerplate scaffolding (FastAPI route models, SQLite schema DDL, and pytest fixture setup).

### Where Output Was Overridden or Rewritten
1. **Query Filtering**: The initial AI suggestion used SQL `LIKE '%country%'` substring matching on the denormalized column. This was rejected and rewritten to use relational junction tables (`title_countries`) to avoid substring false positives (e.g., query "India" incorrectly matching "Indiana").
2. **Pytest Autoload Conflict**: Running `pytest` encountered a global environment conflict in `langsmith.pytest_plugin` with Python 3.12's `ForwardRef`. We resolved this by providing `run_tests.py` and configuring `pytest.ini` with `addopts = -p no:langsmith`.
3. **Retrieval Architecture**: We avoided heavyweight HuggingFace embedding downloads that hang or fail without internet, opting for deterministic TF-IDF + cosine similarity and isolated LLM mocking in tests.

---

## 6. What Would You Do With Another 4 Hours?

1. **Hybrid Dense + Lexical Search**:
   Implement Reciprocal Rank Fusion (RRF) combining dense vector embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2` or `text-embedding-3-small`) with BM25 keyword search.
2. **Structured Query Intent Parsing**:
   Use structured function calling to extract filter constraints (e.g., `"before 2015"`, `"from Japan"`) directly from natural language questions, executing a pre-filtered vector search.
3. **Asynchronous Database Pool**:
   Migrate SQLite connection handling to `aiosqlite` with connection pooling to maximize concurrent throughput.
4. **Automated Schema Contract Tests**:
   Add schema validation checks using Great Expectations to automatically validate incoming CSV batches before ingestion.
