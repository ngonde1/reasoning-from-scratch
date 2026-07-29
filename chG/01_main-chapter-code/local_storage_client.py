import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from chainlit.data.storage_clients.base import BaseStorageClient

class LocalStorageClient(BaseStorageClient):
    def __init__(self, base_dir: str = "uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def upload_file(self, file_path: str = None, data: bytes = None, object_key: str = None, **kwargs) -> dict:
        """
        Save uploaded file into a unique subfolder and return a dict with 'url' and 'object_key'.
        Handles both raw bytes and string data safely.
        """
        original_name = Path(object_key or file_path or "uploaded_file").name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        subfolder = self.base_dir / f"{timestamp}_{unique_id}"
        subfolder.mkdir(parents=True, exist_ok=True)

        dest = subfolder / original_name

        # If raw data is provided
        if data is not None:
            with open(dest, "wb") as f:
                if isinstance(data, str):
                    f.write(data.encode("utf-8"))  # ✅ convert string to bytes
                else:
                    f.write(data)
        # If a file path is provided
        elif file_path and Path(file_path).exists():
            shutil.copy(file_path, dest)

        relative_path = str(dest.relative_to(self.base_dir))
        return {"url": f"/files/{relative_path}", "object_key": relative_path}

    async def get_read_url(self, object_key: str = None, dest_path: str = None, **kwargs) -> str:
        """
        Return a public URL for the stored file.
        Chainlit may call this with object_key or dest_path.
        """
        path = object_key or dest_path
        return f"/files/{path}"

    async def delete_file(self, object_key: str = None, dest_path: str = None, **kwargs) -> None:
        """
        Delete a file by its object key or dest_path.
        """
        path = object_key or dest_path
        if path:
            try:
                os.remove(self.base_dir / path)
                print(f"🗑️ Deleted {self.base_dir / path}")
            except FileNotFoundError:
                print(f"⚠️ File not found: {self.base_dir / path}")

    async def close(self) -> None:
        return
