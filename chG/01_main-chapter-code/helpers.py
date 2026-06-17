import pandas as pd
import psycopg2
import json
import docx
from PyPDF2 import PdfReader
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import chainlit as cl
import time
import threading

# Global flags
cancel_requested = threading.Event()
pause_requested = threading.Event()

def request_cancel():
    cancel_requested.set()

def reset_cancel():
    cancel_requested.clear()

def request_pause():
    pause_requested.set()

def request_resume():
    pause_requested.clear()

def insert_chunk_to_postgres(chunk_rows, source_file, progress=None):
    """Helper to insert one chunk of rows into PostgreSQL with pause/cancel support."""
    if cancel_requested.is_set():
        return

    conn = psycopg2.connect(
        dbname="my_chainlit_db",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    for row in chunk_rows:
        if cancel_requested.is_set():
            break

        # ⏸ Pause support: wait until resume
        while pause_requested.is_set() and not cancel_requested.is_set():
            if progress:
                progress.update(0, label="⏸ Paused")
            time.sleep(0.5)

        # ▶️ Resume label
        if not pause_requested.is_set() and not cancel_requested.is_set() and progress:
            progress.update(0, label="▶️ Resumed")

        for k, v in row.items():
            if isinstance(v, (datetime.date, datetime.datetime)):
                row[k] = v.isoformat()

        row_json = json.dumps(row)
        cur.execute(
            "INSERT INTO file_rows (source_file, row_data) VALUES (%s, %s)",
            (source_file, row_json)
        )

    conn.commit()
    cur.close()
    conn.close()

async def ingest_file_to_postgres(file_path, source_file, chunk_size=1000, max_workers=4):
    """
    Reads Excel, CSV, PDF, DOCX, or TXT files,
    converts each row or text chunk into JSON,
    and inserts into the file_rows table in PostgreSQL.
    Uses chunked + parallel ingestion with Chainlit progress bar,
    showing row counts, ETA, and allowing pause/resume/cancel.
    """

    reset_cancel()
    chunks = []

    # Build chunks by file type
    if file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path).dropna(how="all").reset_index(drop=True)
        df.columns = [str(col) if not str(col).startswith("Unnamed") else f"Column_{i}" for i, col in enumerate(df.columns)]
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start+chunk_size]
            chunks.append([row.dropna().to_dict() for _, row in chunk.iterrows()])

    elif file_path.endswith(".csv"):
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = chunk.dropna(how="all").reset_index(drop=True)
            chunk.columns = [str(col) if not str(col).startswith("Unnamed") else f"Column_{i}" for i, col in enumerate(chunk.columns)]
            chunks.append([row.dropna().to_dict() for _, row in chunk.iterrows()])

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        chunks = [[{"Paragraph": para.text} for para in doc.paragraphs if para.text.strip()]]

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        chunks = [[{"Page": i+1, "Text": page.extract_text()} for i, page in enumerate(reader.pages) if page.extract_text()]]

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            chunks = [[{"Line": line.strip()} for line in f if line.strip()]]

    else:
        raise ValueError("Unsupported file type. Use Excel, CSV, DOCX, PDF, or TXT.")

    # Progress bar
    total_chunks = len(chunks)
    total_rows = sum(len(c) for c in chunks)
    progress = cl.Progress(total=total_chunks, label=f"Ingesting {source_file}...")
    cl.user_session.set("progress_bar", progress)

    inserted_rows = 0
    completed_chunks = 0
    start_time = time.time()

    # Parallel insert
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(insert_chunk_to_postgres, chunk, source_file, progress): (i, len(chunk)) for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            if cancel_requested.is_set():
                progress.update(0, label="⏹ Ingestion cancelled by user.")
                break

            i, chunk_size_done = futures[future]
            future.result()
            inserted_rows += chunk_size_done
            completed_chunks += 1

            # ETA
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / completed_chunks
            remaining_chunks = total_chunks - completed_chunks
            eta_seconds = int(avg_time_per_chunk * remaining_chunks)
            eta_minutes, eta_seconds = divmod(eta_seconds, 60)

            progress.update(
                1,
                label=f"Inserted {inserted_rows} of {total_rows} rows "
                      f"(chunk {i+1} of {total_chunks}) | ETA: {eta_minutes}m {eta_seconds}s"
            )

    if not cancel_requested.is_set():
        await cl.Message(content=f"🎉 Finished inserting {total_rows} rows from {source_file} into file_rows.").send()
    else:
        await cl.Message(content=f"⚠️ Ingestion cancelled after {inserted_rows} rows.").send()
