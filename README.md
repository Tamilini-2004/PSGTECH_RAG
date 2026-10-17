# PSG Tech public-site crawler

This project collects publicly accessible HTML from the approved PSG College
of Technology website, cleans page text, and stores each page with its
normalized source URL in a local SQLite database. The crawler does not download
documents or media. The separate RAG assistant uses a locally running open
source language model through Ollama; it does not require a cloud API key.

## Project files

~~~
PSGTECH_RAG/
├── crawler.py
├── rag_engine.py
├── app.py
├── requirements.txt
├── test_crawler.py
└── README.md
~~~

The database is created as **psgtech_knowledge_base.db** in this project
directory after the initial robots policy check succeeds. It is a persistent
checkpoint: pages committed before a later failure remain available, and
reruns update matching normalized URLs.

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

An HTTP 401, 403, or 429 response stops the run without retrying or attempting
to bypass the restriction.

The crawler does not submit forms, execute JavaScript, retry denied requests,
bypass authentication or CAPTCHAs, or retrieve file and media links. It
extracts a page's title and text from main/article/body content after removing
navigation and other non-content elements. It retains Unicode text, including
Tamil. Pages with fewer than 20 cleaned characters are not stored.

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

## Run the local RAG assistant

The assistant uses the Ollama local model service and Qwen3 4B. Ollama and the
model are installed once on the computer; chat requests then run locally without
an API key. The first model download is about 2.5 GB, and response speed depends
on the computer's available memory and processor or GPU.

1. Install and start [Ollama for Windows](https://ollama.com/download/windows).
2. In PowerShell, download the model:

~~~powershell
ollama pull qwen3:4b
~~~

3. Rebuild the semantic index after the department-aware faculty chunking change:

~~~powershell
.\.venv\Scripts\python.exe .\rag_engine.py
~~~

4. Start the chat application:

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
~~~

The app retrieves multiple semantic and keyword-matched records, including
department-specific faculty roster sections and related programme or course
pages. It can answer follow-up questions using recent conversation context.
When the public records list a department's faculty but do not say which person
handles a particular subject, the assistant reports that limitation instead of
assigning an unsupported specialty. To use a different local model, set the
`OLLAMA_MODEL` environment variable before launching Streamlit; that model must
already be installed in Ollama. For a non-default Ollama host, set
`OLLAMA_BASE_URL`.

## Troubleshooting and ethical use

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
- `rag_engine.py` processes the database into department-aware text chunks and
  a FAISS semantic index for the local RAG assistant.
