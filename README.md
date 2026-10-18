# PSG Tech public-site crawler

This project builds a grounded question-answering assistant from publicly
accessible PSG College of Technology records. A manually started crawler checks
the site's robots policy, fetches approved HTML and optionally PDFs, and stores
cleaned text and source URLs in SQLite. An indexing step creates a persistent
local Chroma semantic-search index. The Streamlit app retrieves relevant
database passages and asks Google's Gemini Developer API to format a grounded
answer. The app reads Gemini key(s) from server environment variables,
Streamlit-managed secrets, or the project `.env` file; it does not ask visitors
to enter a key.

## Project files

~~~
PSGTECH_RAG/
├── crawler.py
├── ingest_documents.py
├── rag_engine.py
├── app.py
├── gemini_retry.py
├── requirements.txt
├── test_crawler.py
└── README.md
~~~

The database is created as **psgtech_knowledge_base.db** in this project
directory after the initial robots policy check succeeds. It is a persistent
checkpoint: pages committed before a later failure remain available, and
reruns update matching normalized URLs.

## End-to-end project flow

The project has four stages. Run them in order; after changing the SQLite
records, rebuild the Chroma index before asking the app about the new content.

1. **Collect records:** run `crawler.py` once by hand. It observes the approved
   PSG Tech hostname list and robots.txt, then saves permitted HTML pages (and
   PDFs only when explicitly enabled) to `psgtech_knowledge_base.db`.
2. **Add a local official PDF if needed:** `ingest_documents.py` extracts its
   selectable text and upserts it under the supplied official PSG Tech URL in
   the same SQLite table. It does not download the PDF.
3. **Build semantic search:** `rag_engine.py` reads the SQLite records and
   writes model-sized text windows and source metadata into the persistent
   `psgtech_chroma_db` Chroma collection. It uses
   `sentence-transformers/all-MiniLM-L6-v2` on CPU; the first index build may
   download the model.
4. **Run chat:** `app.py` loads the existing SQLite records and Chroma index,
   retrieves up to 14 relevant passages (at most two semantic passages per
   source URL), and sends those records to Gemini with grounding instructions.
   Gemini streams a structured JSON answer. The interface displays one
   retrieved, validated source URL and keeps a downloadable session transcript.

Typical full run from the project directory:

~~~powershell
python .\crawler.py --focus-seeds --page-limit 800
python .\rag_engine.py
python -m streamlit run .\app.py
~~~

To include robots-allowed PDFs during the crawl, add `--include-pdfs` to the
crawler command. To add a local PDF instead, use the separate ingestion command
below. After either operation changes the database, run `rag_engine.py` again.
The app does not crawl or rebuild the index automatically when it starts.

The chatbot can only report facts present in the retrieved database passages.
Semantic retrieval can connect related wording and known abbreviations, but it
cannot provide missing source material. A page may be absent, blocked by its
robots policy, hosted on an unapproved external hostname, JavaScript-rendered,
image-only, or published without the requested detail. The one-link answer
format intentionally selects a relevant direct source rather than listing every
retrieved URL.

## Setup on Windows PowerShell

Run these commands from the project directory:

~~~powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v
~~~

If PowerShell blocks environment activation, use this for the current terminal:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

Python 3.10 or newer is required.

## Run the crawl

Start with the defaults:

~~~powershell
python crawler.py
~~~

The default limit is 100 visited page URLs. The request timeout is at most
10 seconds, the minimum delay is one second after each request, and a response
is limited to 2,000,000 bytes. These bounds can be changed from PowerShell:

~~~powershell
python crawler.py --page-limit 50 --max-response-bytes 1000000 --request-delay 2
python crawler.py --database .\data\psgtech_knowledge_base.db
~~~

To prioritize the institution's known programme, faculty, placement, hostel,
and website-directory pages and extract robots-allowed PDFs, opt in explicitly:

~~~powershell
python .\crawler.py --focus-seeds --include-pdfs --page-limit 800
~~~

This remains a single manually started crawl. PDF downloads are bounded to
20 MB each and 500 pages each by default, obey robots.txt, and use the same
approved PSG Tech host list and verified TLS settings. These options do not
guarantee that every requested record is published or extractable.

The delay cannot be set below one second. A larger crawl-delay from robots.txt
is used when present. The crawler uses breadth-first search with a list and
queue index. It does not start crawling when imported as a Python module.

URLs are restricted to **psgtech.edu** and its **www.psgtech.edu** alias, with
the alias normalized to the bare hostname. The seed page must identify itself
as PSG College of Technology. Before fetching pages, the crawler must retrieve
and parse the applicable robots.txt. An HTTP 404 for robots.txt is treated as
no published restrictions, consistent with
[RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html), and uses an explicit
allow-all policy.
Other non-200 responses, redirects to an unapproved host, oversized,
unexpected-content-type, or unreadable robots files stop the crawl. A
robots-disallowed page is skipped. Redirects are examined manually; the crawler
never follows an external redirect.

Pages that link to a separate hostname, such as a library, academic portal, or
placement system, are recorded as links but are not crawled. Each additional
hostname needs its own ownership and robots-policy review before it can be
added. The crawler extracts server-returned HTML and selectable PDF text; it
does not execute JavaScript, OCR scanned PDFs, or infer content from images.
If a placement-statistics page returns only a heading in its HTML, the crawler
cannot produce figures that are absent from that response.

An HTTP 401, 403, or 429 response stops the run without retrying or attempting
to bypass the restriction.

The crawler does not submit forms, execute JavaScript, retry denied requests,
bypass authentication or CAPTCHAs, or retrieve non-PDF file and media links. It
extracts a page's title and text from main/article/body content after removing
navigation and other non-content elements. It retains Unicode text, including
Tamil. Pages with fewer than 20 cleaned characters are not stored.

After the verified homepage is fetched, the one-time crawl checks sitemap URLs
advertised in robots.txt and conventional sitemap paths. Sitemap requests use
the same approved hosts, verified TLS, response-size and timeout limits, and
robots policy checks as page requests. Internal page URLs are added to the BFS
queue; sitemap indexes are followed with a bounded document count.

## Ingest a local PDF

If you already have an official PSG Tech PDF, the standalone ingestion script
can add it to the database without a network request:

~~~powershell
python .\ingest_documents.py `
  .\documents\curriculum.pdf `
  https://psgtech.edu/path/to/curriculum.pdf `
  "Curriculum"
~~~

The script extracts selectable PDF text and upserts it into the existing
`crawled_resources` table using the official source URL. Scanned PDFs without
a text layer are reported and left out; they need OCR before ingestion. Use
`--database .\path\to\psgtech_knowledge_base.db` to select another existing
database. After a crawl or PDF ingestion changes the database, rebuild the RAG
index so the chatbot can retrieve the new records:

~~~powershell
python .\rag_engine.py
~~~

## Example terminal output

Actual page titles, URLs, and counts depend on the site and its robots policy.
A run has this general format:

~~~
[visit 1/100] https://psgtech.edu/
[stored] PSG College of Technology | https://psgtech.edu/
[robots] Disallowed: https://psgtech.edu/example
[visit 2/100] https://psgtech.edu/about
[stored] About the College | https://psgtech.edu/about

Crawl summary
  URLs visited:                    2
  HTML pages successfully fetched: 2
  Pages stored or updated:         2
  Assets/non-HTML skipped:          0
  Robots-disallowed URLs:           1
  Request/database errors:          0
Database: D:\path\to\PSGTECH_RAG\psgtech_knowledge_base.db
~~~

This is an output example, not a report of a crawl run.

## Database schema

The crawled_resources table contains:

| Column | Meaning |
| --- | --- |
| id | SQLite integer primary key |
| url | Unique normalized URL corresponding to the stored page |
| page_title | Extracted HTML title, or NULL if absent |
| sanitized_content | Cleaned visible text |
| extracted_at | Timestamp of the most recent insert or update |
| content_hash | SHA-256 digest of cleaned text |

Values are written with SQLite parameters. An ON CONFLICT(url) DO UPDATE
upsert preserves the row ID when a page is revisited.

## Inspect the database

Use Python's built-in SQLite client from PowerShell:

~~~powershell
@'
import sqlite3

connection = sqlite3.connect("psgtech_knowledge_base.db")
for row in connection.execute(
    """
    SELECT url, page_title, length(sanitized_content) AS content_length
    FROM crawled_resources
    ORDER BY url
    """
):
    print(row)
connection.close()
'@ | python -
~~~

To inspect one stored row, including its source URL and text:

~~~powershell
@'
import sqlite3

connection = sqlite3.connect("psgtech_knowledge_base.db")
row = connection.execute(
    "SELECT url, page_title, sanitized_content FROM crawled_resources LIMIT 1"
).fetchone()
print("URL:", row[0] if row else None)
print("Title:", row[1] if row else None)
print("Content characters:", len(row[2]) if row else 0)
print("Content preview:", row[2][:500] if row else None)
connection.close()
'@ | python -
~~~

The URL and extracted text are both present when a row has a non-empty url
and a positive length(sanitized_content). A database viewer such as DB Browser
for SQLite can also open the .db file directly.

## Run the Gemini RAG assistant

The chat model uses Google's Gemini Developer API with `gemini-3.8-flash`.
Add one or more Google AI Studio API keys to the project-root `.env` without
sharing them:

~~~
GEMINI_API_KEYS=primary_key,backup_key
~~~

`GEMINI_API_KEYS` is a comma-separated list. A single `GEMINI_API_KEY` is still
accepted for backward compatibility. `app.py` loads `.env` with `override=True`,
so the project `.env` value takes precedence over a shell variable of the same
name. Restart Streamlit after changing the credentials. Never commit or share
real keys.

Install the project dependencies in PowerShell:

~~~
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
~~~

Build the persistent Chroma index from the existing SQLite database. The first
build downloads the all-MiniLM-L6-v2 embedding model; it then runs on CPU, so no
GPU is needed. Chroma stores model-sized overlapping search windows and maps
each match back to the full source page in SQLite.

~~~
.\.venv\Scripts\python.exe .\rag_engine.py
~~~

Start Streamlit:

~~~
.\.venv\Scripts\python.exe -m streamlit run .\app.py
~~~

The app streams a grounded JSON response from Gemini and displays a retrieved
database excerpt while generation is in progress. If Gemini fails or returns a
response that does not satisfy the JSON contract, the app displays the closest
retrieved database passage instead of presenting it as an LLM-generated answer.
The local guardrails handle clear gibberish and prompt-injection attempts; the
Gemini prompt handles grounding, ambiguity, and semantic interpretation using
retrieved PSG Tech records. The app uses the official `google-genai` SDK and
does not request the key in the UI. The default model is `gemini-3.8-flash`;
set `GEMINI_MODEL` in the environment if the configured Google API project
requires a different model available to that account.

Gemini requests retry only HTTP 503 errors, using exponential backoff with
random jitter for at most five attempts. Authentication, permission, model, and
other HTTP errors are not retried. Run the local mock example without an API
key or network request:

~~~powershell
.\.venv\Scripts\python.exe .\gemini_retry.py
~~~

Install only the new RAG and model dependencies with:

~~~
.\.venv\Scripts\python.exe -m pip install google-genai langchain-chroma langchain-huggingface sentence-transformers python-dotenv
~~~

## Deploy on Streamlit Community Cloud

The app can be hosted on Streamlit Community Cloud and shared through its
`streamlit.app` URL. Deploy the app source together with
`psgtech_knowledge_base.db` and the complete `psgtech_chroma_db` directory; the
app does not build a missing index on startup. The Chroma files are configured
for Git LFS because the current index contains a file larger than GitHub's
regular Git file limit. Streamlit Community Cloud supports repositories that
use Git LFS.

1. Push the project to a GitHub repository that your account can administer.
   Keep `.env`, `.streamlit/secrets.toml`, and local editor settings out of the
   repository. The `.gitignore` excludes local credentials and `.vscode`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), connect
   GitHub, and create or update the app from the repository. Select branch
   `deploy/streamlit-cloud` and entrypoint `app.py`. The hosted app must use
   this branch to receive the plural Gemini key support in the current code.
3. In the deployment dialog's **Advanced settings**, select a Python version
   compatible with `requirements.txt` and add the secret below. Use the valid
   Google AI Studio Developer API key privately in the settings; do not paste
   it into chat or commit it to GitHub.

   ~~~toml
   GEMINI_API_KEYS = ["primary_key", "backup_key"]
   ~~~

4. Deploy and watch the build and app logs in Streamlit Community Cloud. The
   first app startup downloads the CPU embedding model if the cloud instance
   does not already have it cached. Once startup completes, check a few queries
   and confirm that answers cite the expected PSG Tech source records.

The current project `.env` is for local development only. Cloud secrets are
read from Streamlit's managed secrets store, and the key is never rendered in
the UI. When a request receives a key-specific `401` or `403`, the app tries
the next configured key and reports only the key slot and status code. `429`
quota and `503` service errors do not rotate keys: Google applies Gemini rate
limits per project, so creating extra keys in the same project does not add
quota and rotating projects to evade a limit is not an appropriate fallback.
Use backup credentials only for separately authorized projects with Gemini
access. See Streamlit's official
[deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management),
and [file organization and Git LFS guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Recent errors and troubleshooting

The app has separate retrieval and generation stages. A passage can appear on
screen even when Gemini has not produced an answer: that passage is a database
fallback and is labelled as such. Use the message shown in the app to identify
which stage failed.

| Observed error or symptom | What it means | What to check |
| --- | --- | --- |
| `DH_KEY_TOO_SMALL` while fetching `robots.txt` | The origin's legacy TLS Diffie-Hellman parameters conflict with the crawler's default TLS policy. The crawler must verify robots before fetching pages. | The crawler currently uses a session-scoped TLS compatibility adapter at OpenSSL security level 1, with certificate and hostname verification still enabled and TLS 1.2 or newer required. If the handshake or certificate check still fails, the crawl stops. Ask the PSG Tech site administrator to modernize TLS; do not disable verification or skip robots. |
| Gemini `401 UNAUTHENTICATED`, `API key not valid`, or `ACCESS_TOKEN_TYPE_UNSUPPORTED` | The configured credential is rejected or is not a Google AI Studio Developer API key suitable for the Gemini Developer API. This is separate from database retrieval. | Check that `GEMINI_API_KEY` or `GEMINI_API_KEYS` contains a valid key from an enabled Google AI Studio project and restart Streamlit. Do not paste the key into chat, README, source control, or screenshots. |
| Gemini `403 PERMISSION_DENIED` / project access denied | Google rejected access for the key's project. Changing application code or retrying the same request does not grant project access. | Check the key's project/API status in Google AI Studio/Cloud or contact Google support. Use a key from a project that has Gemini API access. |
| Gemini `404 NOT_FOUND` for a model | The requested model name is unavailable to that API project/account, or the configured model name is invalid. | Check the available model identifier for the key's API. The app defaults to `gemini-3.8-flash`; configure `GEMINI_MODEL` to an identifier supported by the account, then restart Streamlit. |
| Gemini `503 UNAVAILABLE` / high demand | Gemini is temporarily unable to serve the request. | The app retries 503 responses at most five times with bounded exponential backoff and jitter. If all attempts fail, it shows the closest retrieved database passage and labels it as a database excerpt, not a generated answer. |
| Gemini key returns `401` or `403` | That credential is invalid or its project lacks access. | Configure `GEMINI_API_KEYS` as a comma-separated local environment value or a TOML list in Streamlit secrets. The app tries the next key for these key-specific errors. Multiple keys in one Google project share that project's quota; do not use key rotation to evade quota limits. |
| `Gemini returned an invalid structured response` or no response text | The request reached the answer-validation stage, but Gemini returned empty or malformed JSON that did not meet the app's required fields. This is not necessarily an API-key or retrieval failure. | Retry the query. If it persists, inspect the Streamlit error details and the model response format/configuration. The app falls back to retrieved text when available. |
| App stays on “Loading the cached PSG Tech Chroma index and CPU embedding model...” | The app is initializing its local embedding model/index, before it can search or call Gemini. | Build the index once with internet access using `python .\rag_engine.py`; check that `psgtech_chroma_db` and the locally cached embedding model exist, then restart the app. The app startup loads the model locally. |
| Clubs, staff, programmes, or other answers are incomplete | Retrieval can only return records that are present and useful in SQLite/Chroma. Search relevance and context limits can also omit a passage. | Inspect the matching SQLite rows. If needed, crawl permitted PSG Tech pages or ingest an official PDF, then rebuild Chroma with `python .\rag_engine.py`. Do not infer missing facts from a partial excerpt. |
| `No relevant source record` or index missing/empty | The database or persistent Chroma collection has not been built, is empty, or is at a different path. | Run the crawler, then `rag_engine.py`; ensure both `psgtech_knowledge_base.db` and `psgtech_chroma_db` are in the project directory used by Streamlit. |

### Safe operation notes

- If the run stops before crawling, check the printed robots.txt or homepage
  verification error. The crawler intentionally fails closed when it cannot
  verify permission.
- If a proxy, firewall, or network outage prevents requests, resolve that
  connectivity issue and rerun. Previously committed pages remain in SQLite.
- The crawler mounts a session-scoped TLS compatibility adapter at OpenSSL
  security level 1 to accommodate the current `DH_KEY_TOO_SMALL` server error.
  It keeps certificate and hostname verification enabled and requires TLS 1.2 or
  newer. Level 1 accepts weaker Diffie-Hellman parameters, so this is a temporary
  compatibility tradeoff for the educational crawl, not a production setting.
  If robots.txt still cannot be retrieved, the crawler stops before requesting
  pages. Ask the site administrator to update TLS, then remove the compatibility
  adapter and return to the default OpenSSL security level.
- Increase --page-limit gradually and keep the delay at or above one second.
- The crawl does not include external PSG-related sites because they are
  separate hostnames. Add a host only after separately confirming permission
  and updating the allowlist and its robots handling.
- Review the target site's current terms and robots.txt before each crawl.
  Public accessibility alone does not grant permission to ignore site rules.
- `rag_engine.py` reads the existing SQLite schema without changing it. It
  preserves long structured faculty, programme, course, and syllabus pages as
  logical records, then splits them into smaller embedding windows for Chroma.
  The SQLite page text remains the source of truth.
- Ollama is not an active fallback in the current app. Gemini failures produce
  a clearly labelled retrieved-database excerpt when relevant records exist;
  that excerpt is not a generated semantic answer.
- Keep `.env` private. `.gitignore` excludes it. If a key has been pasted into a
  chat, committed, or otherwise exposed, revoke it and create a replacement.
