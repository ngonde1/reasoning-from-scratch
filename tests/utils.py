from pathlib import Path              # Import Path to handle filesystem paths in a clean, cross‑platform way
import sys                            # Import sys to write progress updates directly to the terminal
import requests                       # Import requests library to make HTTP requests for downloading files
from urllib.parse import urlparse     # Import urlparse to extract filename from a URL

def _download_error_message(filename, url, primary_error, backup_url=None, backup_error=None):   # Define helper to build detailed error messages
    details = [f"Failed to download {filename}."]   # Start error message list with filename info

    if primary_error is not None:                   # If the primary URL failed
        details.append(                            # Add details about the primary error
            f"Primary URL failed ({url}): "
            f"{type(primary_error).__name__}: {primary_error}"
        )

    if backup_url and backup_error is not None:    # If a backup URL was tried and also failed
        details.append(                            # Add details about the backup error
            f"Backup URL failed ({backup_url}): "
            f"{type(backup_error).__name__}: {backup_error}"
        )

    cert_or_proxy_issue = any(                     # Check if errors are related to SSL or proxy issues
        isinstance(err, (requests.exceptions.ProxyError, requests.exceptions.SSLError))
        for err in (primary_error, backup_error)
        if err is not None
    )
    if not cert_or_proxy_issue:                    # If not already flagged, check error text for keywords
        lowered = " ".join(
            str(err).lower() for err in (primary_error, backup_error) if err is not None
        )
        cert_or_proxy_issue = any(
            keyword in lowered for keyword in ("certificate", "ssl", "tls", "proxy")
        )

    if cert_or_proxy_issue:                        # If SSL/proxy issue detected, add explanation
        details.append(
            "This can happen on work or school machines where a VPN, proxy, or "
            "antivirus tool intercepts HTTPS certificates."
        )

    details.append(                                # Add link to troubleshooting guide
        "See the troubleshooting guide: "
        "https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md "
        "(especially the 'File Download Issues' section)."
    )
    return "\n".join(details)                      # Return full error message as a string

def download_file(url, out_dir=".", backup_url=None):   # Main function to download a file
    out_dir = Path(out_dir)                             # Convert output directory to Path object
    out_dir.mkdir(parents=True, exist_ok=True)          # Create directory if it doesn’t exist
    filename = Path(urlparse(url).path).name            # Extract filename from URL
    dest = out_dir / filename                           # Full destination path for file

    def try_download(u):                                # Inner helper function to attempt download
        try:
            with requests.get(u, stream=True, timeout=30) as r:   # Make HTTP request with timeout
                r.raise_for_status()                               # Raise error if status not 200
                size_remote = int(r.headers.get("Content-Length", 0))  # Get file size from server

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")          # File already matches remote size
                    return True, None

                # Download in 1 MiB chunks with progress display
                block = 1024 * 1024                                # Define chunk size (1 MiB)
                downloaded = 0                                     # Track bytes downloaded
                with open(dest, "wb") as f:                        # Open file for writing
                    for chunk in r.iter_content(chunk_size=block): # Iterate over chunks
                        if not chunk: continue                     # Skip empty chunks
                        f.write(chunk)                             # Write chunk to file
                        downloaded += len(chunk)                   # Update progress
                        if size_remote:                            # If size known, show progress %
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / "
                                f"{size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote: sys.stdout.write("\n")             # Print newline after progress
            return True, None                                      # Success
        except requests.RequestException as exc:                   # Catch request errors
            return False, exc                                      # Return failure and error

    # Try main URL first
    success, primary_error = try_download(url)                     # Attempt primary download
    if success: return dest                                        # Return file path if successful

    # Try backup URL if provided
    backup_error = None
    if backup_url:                                                 # If backup URL exists
        print(f"Primary URL ({url}) failed.\nTrying backup URL ({backup_url})...")
        success, backup_error = try_download(backup_url)           # Attempt backup download
        if success: return dest                                    # Return file path if successful

    message = _download_error_message(                             # Build detailed error message
        filename=filename,
        url=url,
        primary_error=primary_error,
        backup_url=backup_url,
        backup_error=backup_error,
    )
    raise RuntimeError(message) from (backup_error or primary_error)   # Raise error with context
