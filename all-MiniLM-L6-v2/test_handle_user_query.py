import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer, models
import faiss

# Import your functions from the main app file
from qwen3_chat_interface_multiturn import handle_user_query, prepare_file_context

async def run_tests():
    # Simulate a small CSV file
    file_path = "sample.csv"
    headers = ["Name", "Age", "Role"]
    file_table = [
        {"Name": "Alice", "Age": 30, "Role": "Engineer"},
        {"Name": "Bob", "Age": 25, "Role": "Designer"},
        {"Name": "Charlie", "Age": 35, "Role": "Manager"},
    ]
    file_text = "\n".join(" | ".join(f"{k}: {v}" for k, v in r.items()) for r in file_table)

    # Test 1: Row count
    answer_rows = await handle_user_query("How many rows are in this file?", file_table, headers, file_text, file_path)
    print("Test 1 (rows):", answer_rows)

    # Test 2: Column count
    answer_cols = await handle_user_query("How many columns are in this file?", file_table, headers, file_text, file_path)
    print("Test 2 (columns):", answer_cols)

    # Test 3: FAISS fallback (semantic query)
    # Build FAISS index from sample text
    index, texts, st_model, lang_code = await prepare_file_context(file_path, headers)
    query = "Who is the manager?"
    answer_faiss = await handle_user_query(query, file_table, headers, file_text, file_path)
    print("Test 3 (FAISS fallback):", answer_faiss)

if __name__ == "__main__":
    asyncio.run(run_tests())
