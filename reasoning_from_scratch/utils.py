

from pathlib import Path
import sys
import requests
from urllib.parse import urlparse

import os, json, datetime
from sqlalchemy import text as sql_text
import chainlit as cl

def sanitize_row(row: dict) -> dict:
    """Convert non-serializable values (like datetime) into strings."""
    clean_row = {}
    for k, v in row.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            clean_row[k] = v.isoformat()
        elif v is None:
            clean_row[k] = ""
        else:
            clean_row[k] = v
    return clean_row

def chunk_rows(rows, chunk_size=500):
    """Split large tabular datasets into manageable chunks."""
    for i in range(0, len(rows), chunk_size):
        yield rows[i:i+chunk_size]

def chunk_text(text: str, chunk_size=1000, overlap=100):
    """Split raw text into overlapping chunks for FAISS indexing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_file(file_path: str):
    """Dispatcher to load files by extension."""
    ext = Path(file_path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        from .file_ingestion import load_excel
        return load_excel(file_path)
    elif ext == ".csv":
        from .file_ingestion import load_csv
        return load_csv(file_path)
    elif ext == ".docx":
        from .file_ingestion import load_docx
        return [], load_docx(file_path)
    elif ext == ".pdf":
        from .file_ingestion import load_pdf
        return [], load_pdf(file_path)
    elif ext == ".txt":
        from .file_ingestion import load_txt
        return [], load_txt(file_path)
    elif ext == ".json":
        from .file_ingestion import load_json
        return load_json(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [], [{"RawText": f.read()}]

def get_data_layer():
    """Return SQLAlchemy data layer with LocalStorageClient."""
    conninfo = os.getenv("DATABASE_URL")
    if not conninfo:
        raise ValueError("DATABASE_URL not found in environment variables.")
    from local_storage_client import LocalStorageClient
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    return SQLAlchemyDataLayer(
        conninfo=conninfo,
        storage_provider=LocalStorageClient(base_dir="uploads")
    )


def download_file(url, out_dir=".", backup_url=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u):
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True
        except requests.RequestException:
            return False

    # Try main URL first
    if try_download(url):
        return dest

    # Try backup URL if provided
    if backup_url:
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        if try_download(backup_url):
            return dest

    raise RuntimeError(f"Failed to download {filename} from both mirrors.")
