# --- Standard library imports ---
import os
import asyncio
import time
import csv
import re
from pathlib import Path
from typing import TypedDict, List
import langcodes

# Toggle streaming mode
USE_STREAMING = True   # for live typing
# USE_STREAMING = False  # for full polished answer

# --- Third-party libraries ---
import torch
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from mcp import ClientSession  # ✅ MCP import
import socketio  # ✅ Added for payload limit
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
from dotenv import load_dotenv
load_dotenv()


# --- Chainlit elements ---
from chainlit import Image, Pdf

# --- File handling libraries ---
import PyPDF2
import docx
import openpyxl
from PIL import Image as PILImage
import pytesseract
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Local project imports ---
from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer, load_tokenizer_only
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

# ------------------- Socket.IO Payload Limit -------------------
# Increase payload limit to 200 MB
sio = socketio.AsyncServer(
    async_mode="asgi",
    max_http_buffer_size=200 * 1024 * 1024  # 200 MB
)

# ------------------- Configuration -------------------
WHICH_MODEL = "reasoning"
MAX_NEW_TOKENS = 512
LOCAL_DIR = "qwen3"
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False

DEVICE = get_device()

def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    tokenizer = load_tokenizer_only(which_model=WHICH_MODEL, local_dir=LOCAL_DIR)
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)
    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)
    return model, tokenizer

MODEL, TOKENIZER = load_app_model_and_tokenizer()

EOS_TOKEN_IDS = (
    TOKENIZER.encode("<|im_end|>")[0],
    TOKENIZER.encode("<|endoftext|>")[0]
)

# ------------------- Agent State -------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]

# ------------------- Sync Qwen3 runner -------------------
def run_qwen_sync(prompt: str) -> str:
    input_ids = TOKENIZER.encode(prompt)
    max_context = MODEL.cfg["context_length"]

    # Truncate safely to avoid exceeding context length
    if len(input_ids) > max_context:
        input_ids = input_ids[-max_context:]

    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    output = ""
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,   # <-- pass raw IDs, not embeddings
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0)
        if token_id in EOS_TOKEN_IDS:
            break
        output += TOKENIZER.decode(token_id.tolist())
    return output

# ------------------- Authentication & Data Layer -------------------
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    return None

@cl.data_layer
def get_data_layer():
    conninfo = os.getenv("DATABASE_URL")
    if not conninfo:
        raise ValueError("DATABASE_URL not found in environment variables.")
    return SQLAlchemyDataLayer(conninfo=conninfo)

# ------------------- Stripe Toggle -------------------
def register_tools(app):
    if os.getenv("STRIPE_ENABLED", "false").lower() == "true":
        from stripe_mcp import StripeMCP
        app.register_tool(StripeMCP())
        print("✅ Stripe MCP enabled.")
    else:
        print("🚫 Stripe MCP disabled — prompt-only mode.")

# ------------------- MCP Integration -------------------
@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
    } for t in result.tools]
    cl.user_session.set("mcp_tools", {connection.name: tools})
    print(f"✅ Connected to {connection.name} with tools: {[t['name'] for t in tools]}")

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    print(f"❌ MCP connection {name} closed.")

@cl.step(type="tool")
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = next((name for name, tools in mcp_tools.items() if any(t["name"] == tool_name for t in tools)), None)
    if not mcp_name:
        return {"error": f"Tool {tool_name} not found"}
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    result = await mcp_session.call_tool(tool_name, tool_input)
    return result

# ------------------- Chat Lifecycle -------------------
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    try:
        messages = []
        for m in thread.get("messages", []):
            role = m.get("role")
            content = m.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        cl.user_session.set("state", {"messages": messages})
    except Exception as e:
        print(f"\nError resuming chat: {e}")
        cl.user_session.set("state", {"messages": []})

@cl.on_chat_start
async def on_chat_start():
    MODEL.reset_kv_cache()

def chunk_rows(rows, chunk_size=500):
    """Split large tabular datasets into manageable chunks."""
    for i in range(0, len(rows), chunk_size):
        yield rows[i:i+chunk_size]

def load_excel(file_path):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    headers = []
    for i, cell in enumerate(next(sheet.iter_rows(values_only=True))):
        if cell and str(cell).strip():
            headers.append(str(cell).strip())
        else:
            headers.append(f"Column{i}")  # fallback only if truly empty
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        row_dict = {headers[i]: row[i] for i in range(len(headers))}
        rows.append(row_dict)
    return headers, rows

def load_csv(file_path):
    rows, headers = [], []
    with open(file_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
    return headers, rows

def load_docx(file_path):
    doc = docx.Document(file_path)
    texts = [para.text for para in doc.paragraphs if para.text.strip()]
    return texts

def load_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return texts

def auto_detect_column(headers, user_message):
    msg_lower = user_message.lower()
    for h in headers:
        if h.lower() in ["project name", "department"]:
            return h
    for h in headers:
        if h.lower() in msg_lower:
            return h
    return headers[0] if headers else None

from difflib import get_close_matches

def fuzzy_detect_column(headers, user_message, cutoff=0.6):
    """
    Fuzzy column detector for a single file:
    - Matches user query words to headers even if not exact.
    - Example: "Sale" -> "Sales", "Dept" -> "Department".
    Works with ANY file headers (Revenue, ClientName, Budget2026, etc.).
    """
    msg_lower = user_message.lower()
    # Try exact match first
    for h in headers:
        if h.lower() in msg_lower:
            return h

    # Fuzzy match using difflib
    for word in msg_lower.split():
        matches = get_close_matches(word, headers, n=1, cutoff=cutoff)
        if matches:
            return matches[0]

    # Fallback: first header if nothing matches
    return headers[0] if headers else None

def fuzzy_detect_column_multi(file_headers_dict, user_message, cutoff=0.6):
    """
    Fuzzy column detector across multiple files:
    - file_headers_dict: {filename: [headers]}
    - Returns a unified mapping of user query -> best header per file.
    Example: {"file1.xlsx": ["Dept"], "file2.csv": ["Department"]}
             query "department" -> {"file1.xlsx": "Dept", "file2.csv": "Department"}
    """
    unified_mapping = {}
    msg_lower = user_message.lower()

    for fname, headers in file_headers_dict.items():
        # Try exact match first
        matched = None
        for h in headers:
            if h.lower() in msg_lower:
                matched = h
                break

        # Fuzzy match if no exact match
        if not matched:
            for word in msg_lower.split():
                matches = get_close_matches(word, headers, n=1, cutoff=cutoff)
                if matches:
                    matched = matches[0]
                    break

        # Fallback: first header if nothing matches
        if not matched and headers:
            matched = headers[0]

        unified_mapping[fname] = matched

    return unified_mapping


def extract_keywords(user_message, headers, rows):
    msg_lower = user_message.lower()
    found = set()
    for h in headers:
        for r in rows:
            val = str(r.get(h, "")).strip().lower()
            if val and val in msg_lower:
                found.add(val)
    if found:
        return list(found)
    msg_parts = msg_lower.split()
    return [" ".join(msg_parts[-3:]).strip()]

def filter_rows_by_values(rows, column_name, values):
    results = []
    for r in rows:
        cell_val = str(r.get(column_name, "")).strip().lower()
        if any(v in cell_val for v in values):
            results.append(r)
    return results

def parse_numeric(value):
    """
    Robust universal numeric parser:
    - Handles strings like '$50,994', '€150,000', '12%', 'Age: 45', 'EUR 2,500', '₦1,200,000'
    - Strips currency symbols, commas, %, text noise
    - Returns float if parsing succeeds, else None
    """
    if value is None:
        return None

    text = str(value).strip()

    # Remove common currency symbols and unit labels
    text = re.sub(r"(USD|EUR|GBP|JPY|CFA|NGN|INR|CAD|AUD|CHF|₦|₿|¥|€|£|\$)", "", text, flags=re.IGNORECASE)

    # Remove commas, percent signs, and extra words
    text = re.sub(r"[,%]", "", text)
    text = re.sub(r"[A-Za-z:]", "", text)  # strip stray letters like 'Age:' or 'Salary'

    # Keep only digits, dot, minus
    cleaned = re.sub(r"[^\d\.\-]", "", text)

    try:
        return float(cleaned)
    except ValueError:
        return None


def filter_rows_by_numeric(rows, column_name, threshold, operator=">"):
    """
    Universal numeric filter for ANY file type:
    - rows: list of dicts (from Excel, CSV, JSON, PDF text, DOCX tables, etc.)
    - column_name: header to check (e.g., 'Annual Salary')
    - threshold: numeric threshold (e.g., 150000)
    - operator: comparison ('>', '<', '>=', '<=', '==')

    Returns only rows that satisfy the condition.
    """
    results = []
    for r in rows:
        val = parse_numeric(r.get(column_name))
        if val is None:
            continue

        if operator == ">" and val > threshold:
            results.append(r)
        elif operator == "<" and val < threshold:
            results.append(r)
        elif operator == ">=" and val >= threshold:
            results.append(r)
        elif operator == "<=" and val <= threshold:
            results.append(r)
        elif operator == "==" and val == threshold:
            results.append(r)

    return results

def get_filtered_table(rows, column_name, values, headers=None):
    filtered = filter_rows_by_values(rows, column_name, values)
    if not headers and filtered:
        headers = list(filtered[0].keys())
    if not filtered:
        return "No results found."
    # Build Markdown table
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in filtered:
        row_values = [str(r.get(h, "")) for h in headers]
        table += "| " + " | ".join(row_values) + " |\n"
    return table

def build_table(rows, headers, limit=20):
    if not rows:
        return "⚠️ No matching records found."
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows[:limit]:
        row_values = [str(r.get(h, "")) for h in headers]
        table += "| " + " | ".join(row_values) + " |\n"
    return table

def compute_all_numeric_stats(rows: list, headers: list) -> str:
    """
    Auto-detect all numeric columns in any file table (Excel, CSV, JSON, etc.)
    and compute sum, average, min, max for each.
    """
    numeric_stats = {}

    for col in headers:
        values = []
        for r in rows:
            val = r.get(col)
            try:
                num = float(val)
                values.append(num)
            except (TypeError, ValueError):
                continue

        if values:
            numeric_stats[col] = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    if not numeric_stats:
        return "⚠️ No numeric columns found to compute stats."

    # Build Markdown summary
    result = "📊 **Numeric Stats for All Columns:**\n\n"
    for col, stats in numeric_stats.items():
        result += (
            f"- Column **{col}**:\n"
            f"  • Count: {stats['count']}\n"
            f"  • Sum: {stats['sum']:.2f}\n"
            f"  • Average: {stats['avg']:.2f}\n"
            f"  • Min: {stats['min']:.2f}\n"
            f"  • Max: {stats['max']:.2f}\n\n"
        )
    return result

import re

def detect_conditions(user_message: str, headers: list):
    """
    Detects numeric and string conditions from user query.
    Supports AND / OR logic.
    Returns a list of (column_name, value/threshold, operator, logic).
    """

    msg_lower = user_message.lower()

    # Map words to operators
    operator_map = {
        "above": ">",
        "greater than": ">",
        "over": ">",
        "below": "<",
        "less than": "<",
        "under": "<",
        "at least": ">=",
        "minimum": ">=",
        "not less than": ">=",
        "at most": "<=",
        "maximum": "<=",
        "not more than": "<=",
        "equal to": "==",
        "equals": "==",
        "exactly": "==",
        "=": "=="
    }

    conditions = []
    parts = re.split(r"\b(and|or)\b", msg_lower)

    logic = "AND"  # default
    for part in parts:
        part = part.strip()
        if part == "and":
            logic = "AND"
            continue
        elif part == "or":
            logic = "OR"
            continue

        # Detect operator
        operator = None
        for phrase, op in operator_map.items():
            if phrase in part:
                operator = op
                break

        # Detect numeric threshold
        match = re.search(r"(\d[\d,\.]*)", part)
        threshold = None
        if match:
            threshold = float(match.group(1).replace(",", ""))

        # Detect column name by fuzzy match
        column_name = None
        for h in headers:
            if h.lower() in part:
                column_name = h
                break

        # String condition (e.g., Department = Finance)
        if operator == "==" and not threshold:
            # Extract word after '=' or 'equals'
            str_match = re.search(r"(?:=|equals)\s+([a-zA-Z0-9\s]+)", part)
            if str_match and column_name:
                value = str_match.group(1).strip()
                conditions.append((column_name, value, "==", logic))
                continue

        # Numeric condition
        if operator and threshold and column_name:
            conditions.append((column_name, threshold, operator, logic))

    return conditions


def apply_conditions(rows: list, conditions: list):
    """
    Apply multiple numeric and string conditions with AND/OR logic.
    """
    if not conditions:
        return rows

    # Start with first condition
    col, val, op, _ = conditions[0]
    if isinstance(val, (int, float)):
        filtered = filter_rows_by_numeric(rows, col, val, operator=op)
    else:
        filtered = filter_rows_by_values(rows, col, [val])

    for col, val, op, logic in conditions[1:]:
        if isinstance(val, (int, float)):
            next_filtered = filter_rows_by_numeric(rows, col, val, operator=op)
        else:
            next_filtered = filter_rows_by_values(rows, col, [val])

        if logic == "AND":
            # Intersection
            filtered = [r for r in filtered if r in next_filtered]
        elif logic == "OR":
            # Union
            filtered = list({id(r): r for r in (filtered + next_filtered)}.values())

    return filtered


async def professional_llm_response(user_query: str, file_table: list, headers: list, llm_runner) -> str:
    """
    Final wrapper:
    - Auto-detects numeric AND string conditions
    - Supports AND/OR logic
    - Applies combined filters before summarizing
    - Adds domain-adaptive enrichment (Finance, HR, Legal, Project, Customer, Compliance)
    - Always enforces professional format (summary → table → insights → closing note)
    """

    # Step 1: Apply conditions
    filtered_rows = file_table
    conditions = detect_conditions(user_query, headers)
    if conditions:
        filtered_rows = apply_conditions(file_table, conditions)

    # Step 2: Convert filtered rows into text for LLM
    if filtered_rows:
        table_text = "\n".join([
            " | ".join(f"{k}: {v}" for k, v in row.items())
            for row in filtered_rows[:50]  # limit to 50 rows for prompt efficiency
        ])
    else:
        table_text = "No matching records found."

    # Step 3: Compute numeric stats
    numeric_stats = compute_all_numeric_stats(filtered_rows, headers) if filtered_rows else ""

    # Step 4: Domain-adaptive enrichment
    enrichment = ""
    st_model = cl.user_session.get("st_model")
    texts = cl.user_session.get("file_text", "").split("\n")
    if st_model and texts:
        domain = await detect_file_domain(texts, st_model)

        if domain == "Finance":
            total_revenue = sum(float(r.get("Revenue", 0) or 0) for r in filtered_rows)
            total_expenses = sum(float(r.get("Expenses", 0) or 0) for r in filtered_rows)
            margin = (total_revenue - total_expenses) / total_revenue * 100 if total_revenue else 0
            enrichment = (
                f"\n💰 **Finance Insight:**\n"
                f"- Profit Margin: {margin:.2f}%\n"
                f"- Total Revenue: {total_revenue:.2f}\n"
                f"- Total Expenses: {total_expenses:.2f}"
            )

        elif domain == "HR":
            total_employees = len(filtered_rows)
            leavers = sum(1 for r in filtered_rows if str(r.get("Status", "")).lower() in ["left", "terminated", "resigned"])
            turnover = (leavers / total_employees * 100) if total_employees else 0
            enrichment = (
                f"\n👥 **HR Insight:**\n"
                f"- Total Employees: {total_employees}\n"
                f"- Leavers: {leavers}\n"
                f"- Turnover Rate: {turnover:.2f}%"
            )

        elif domain == "Legal":
            clause_count = sum(1 for r in filtered_rows if r.get("Clause"))
            enrichment = (
                f"\n⚖️ **Legal Insight:**\n"
                f"- Total Clauses Detected: {clause_count}\n"
                f"- Compliance Risks may need review."
            )

        elif domain == "Project":
            enrichment = (
                f"\n📈 **Project Insight:**\n"
                f"- Milestones tracked: {len(filtered_rows)}\n"
                f"- Check deadlines for risk."
            )

        elif domain == "Customer":
            enrichment = (
                f"\n🤝 **Customer Insight:**\n"
                f"- Orders/Feedback records: {len(filtered_rows)}\n"
                f"- Retention trends visible."
            )

        elif domain == "Compliance":
            enrichment = (
                f"\n🛡️ **Compliance Insight:**\n"
                f"- Policy adherence checks\n"
                f"- Risk factors detected."
            )

    # Step 5: Build prompt for LLM
    prompt = f"""
You are a professional data analyst.
You are given a user query and extracted file content.
Respond in a polished, professional format that ALWAYS follows this structure:

1. Executive Summary
2. Representative Table
3. Narrative Insights
4. Closing Note

Rules:
- Never hard-code words from a specific file; adapt dynamically
- Always keep the tone professional, clear, and insightful
- Always follow the format above, even if the file is text-based or unstructured

User Query:
{user_query}

Filtered Data (sample):
{table_text}

Numeric Stats:
{numeric_stats}

Domain Enrichment:
{enrichment}
    """

    return await llm_runner(prompt)

async def translate_text(text: str, target_lang: str) -> str:
    """
    Simple translation helper using Qwen3 itself.
    """
    prompt = f"Translate the following text into {target_lang}:\n\n{text}"
    return await cl.make_async(run_qwen_sync)(prompt)

# ✅ Insert here
async def summarize_chunk(chunk: str, lang: str) -> str:
    """Summarize a single chunk asynchronously."""
    prompt = f"Summarize this chunk in {lang}:\n{chunk}"
    summary = await cl.make_async(run_qwen_sync)(prompt)
    return f"🔹 {summary}"

async def summarize_file_text(file_text: str, lang: str = "English") -> str:
    """
    Parallel summarization for large text files.
    Splits text into chunks and runs all summaries concurrently.
    """
    chunk_size = 4000
    chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]

    # ✅ Run all summarizations in parallel
    tasks = [summarize_chunk(chunk, lang) for chunk in chunks[:10]]
    summaries = await asyncio.gather(*tasks)

    return f"📄 **Summary (in {lang}):**\n\n" + "\n".join(summaries)

async def handle_multi_file_query(user_query: str, files: list, use_multi_fuzzy: bool = True) -> str:
    """
    Cross-file reasoning handler with optional multi-file fuzzy detection:
    - Merges tables and text from multiple files
    - Runs numeric stats for structured data
    - Summarizes unstructured text
    - Applies domain-specific enrichment automatically
    - Optionally unifies similar headers across files (Dept vs Department)
    """

    merged_tables = []
    merged_headers = set()
    merged_texts = []
    file_headers_dict = {}

    # Collect data from all files
    for f in files:
        headers = cl.user_session.get(f"{f}_headers", [])
        rows = cl.user_session.get(f"{f}_table", [])
        text = cl.user_session.get(f"{f}_text", "")

        if headers:
            file_headers_dict[f] = headers
        if rows:
            merged_tables.extend(rows)
            merged_headers.update(headers)
        if text:
            merged_texts.append(text)

    # 🔹 Fuzzy detection (optional)
    unified_cols = {}
    if use_multi_fuzzy and file_headers_dict:
        unified_cols = fuzzy_detect_column_multi(file_headers_dict, user_query)
    else:
        if merged_headers:
            unified_cols = {files[0]: fuzzy_detect_column(list(merged_headers), user_query)}

    # 🔹 Numeric stats across all files
    stats_summary = ""
    if merged_tables and merged_headers:
        stats_summary = compute_all_numeric_stats(merged_tables, list(merged_headers))

    # 🔹 Summarize text across all files
    text_summary = ""
    if merged_texts:
        combined_text = "\n".join(merged_texts)
        text_summary = await summarize_file_text(combined_text, lang="English")

    # 🔹 Domain-specific enrichment
    enrichment = ""
    st_model = cl.user_session.get("st_model")
    for f in files:
        texts = cl.user_session.get(f"{f}_text", "").split("\n")
        if texts and st_model:
            domain = await detect_file_domain(texts, st_model)
            if domain == "Finance":
                enrichment += "\n💰 **Finance Insights:**\n- Profit margin analysis\n- Growth trends\n"
            elif domain == "Legal":
                enrichment += "\n⚖️ **Legal Insights:**\n- Clause extraction\n- Compliance risks\n"
            elif domain == "HR":
                enrichment += "\n👥 **HR Insights:**\n- Salary distribution\n- Turnover detection\n"
            elif domain == "Project":
                enrichment += "\n📈 **Project Insights:**\n- Milestone tracking\n- Deadline risks\n"
            elif domain == "Compliance":
                enrichment += "\n🛡️ **Compliance Insights:**\n- Policy adherence\n- Risk factors\n"
            elif domain == "Customer":
                enrichment += "\n🤝 **Customer Insights:**\n- Order trends\n- Feedback themes\n"

    # 🔹 Build unified answer
    final_answer = "📂 **Cross‑File Unified Answer:**\n\n"
    if unified_cols:
        final_answer += f"🔎 **Unified Column Mapping:** {unified_cols}\n\n"
    if stats_summary:
        final_answer += stats_summary + "\n"
    if text_summary:
        final_answer += text_summary + "\n"
    if enrichment:
        final_answer += enrichment + "\n"

    if not stats_summary and not text_summary and not enrichment:
        final_answer += "⚠️ No usable data found across files."

    return final_answer

async def detect_file_domain(texts: list, st_model) -> str:
    """
    Detects the domain of a file using embeddings + similarity classification.
    Domains: Finance, Legal, HR, Project, Compliance, Customer
    """

    # Candidate domain labels
    domains = {
        "Finance": ["revenue", "sales", "profit", "expenses", "budget"],
        "Legal": ["contract", "agreement", "clause", "obligation", "law"],
        "HR": ["employee", "salary", "turnover", "department", "hiring"],
        "Project": ["project", "milestone", "deadline", "progress", "task"],
        "Compliance": ["policy", "audit", "compliance", "risk", "regulation"],
        "Customer": ["customer", "client", "order", "feedback", "retention"]
    }

    # Encode sample text
    sample = " ".join(texts[:5])  # take first few chunks
    sample_vec = st_model.encode([sample])

    # Encode domain keywords
    scores = {}
    for domain, keywords in domains.items():
        keyword_vec = st_model.encode([" ".join(keywords)])
        similarity = float(np.dot(sample_vec, keyword_vec.T) / (np.linalg.norm(sample_vec) * np.linalg.norm(keyword_vec)))
        scores[domain] = similarity

    # Pick best match
    best_domain = max(scores, key=scores.get)
    return best_domain


async def handle_user_query(
    msg_content: str,
    file_table: list,
    headers: list,
    file_text: str,
    file_path: str
):
    """
    Unified query handler:
    - Uses actual Excel headers (no ColumnX fallback)
    - Applies numeric/string filters
    - Builds a clean Representative Table with build_table()
    - Drops language detection completely
    """

    # If tabular data is available → structured response
    if file_table and headers:
        # Apply conditions
        conditions = detect_conditions(msg_content, headers)
        filtered_rows = apply_conditions(file_table, conditions) if conditions else file_table

        # ✅ Use build_table instead of manual stitching
        table = build_table(filtered_rows, headers, limit=20)

        # Numeric stats
        numeric_stats = compute_all_numeric_stats(filtered_rows, headers) if filtered_rows else ""

        # Build professional response
        response = f"""
1. **Executive Summary**
Query: {msg_content}
Rows matched: {len(filtered_rows)}

2. **Representative Table**
{table}

3. **Narrative Insights**
{numeric_stats}

4. **Closing Note**
Analysis complete. Data shown above is based on actual Excel headers.
"""
        return response

    # Fallback: semantic search
    index = cl.user_session.get("file_index")
    st_model = cl.user_session.get("st_model")
    texts = cl.user_session.get("file_text", "").split("\n")

    if index and st_model and texts:
        query_vec = st_model.encode([msg_content])
        D, I = index.search(np.array(query_vec), k=3)
        if len(I[0]) > 0:
            context = "\n".join([texts[i] for i in I[0]])
            prompt = f"User question: {msg_content}\n\nRelevant context:\n{context}\n\nPlease answer clearly."
            llm_answer = await cl.make_async(run_qwen_sync)(prompt)
            return f"📝 **Answer:**\n\n{llm_answer}"

    # Fallback: summarization
    if file_text:
        return await summarize_file_text(file_text, "English")

    return "⚠️ No data available to answer your query."

async def build_faiss_index(texts):
    word_embedding_model = modules.Transformer("all-MiniLM-L6-v2")
    pooling_model = modules.Pooling(word_embedding_model.get_embedding_dimension())
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    batch_size = 200
    total_batches = (len(texts) + batch_size - 1) // batch_size

    embeddings_list = []
    for i in range(total_batches):
        batch = texts[i*batch_size:(i+1)*batch_size]

        # ✅ Console log only
        print(f"Processing batch {i+1}/{total_batches} in PowerShell...")

        # Run asynchronously to avoid blocking
        batch_embeddings = await cl.make_async(st_model.encode)(batch, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, st_model

async def prepare_file_context(file_path: str):
    """
    Clean ingestion helper:
    - Handles Excel, CSV, DOCX, PDF, TXT, JSON, and images.
    - Builds FAISS index for semantic search.
    - No language detection.
    """

    MODEL.reset_kv_cache()
    file_path = os.path.abspath(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    headers, rows, texts = [], [], []

    # File type handling
    if ext in [".xls", ".xlsx"]:
        headers, rows = load_excel(file_path)
    elif ext == ".csv":
        headers, rows = load_csv(file_path)
    elif ext == ".docx":
        texts = load_docx(file_path)
    elif ext == ".pdf":
        texts = load_pdf(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texts = f.readlines()
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = PILImage.open(file_path)
        texts = [pytesseract.image_to_string(img)]
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [json.dumps(data, indent=2)]
    else:
        texts = [f"Unsupported file type: {ext}"]

    # Chunk rows for tabular files
    if rows:
        texts = []
        for chunk in chunk_rows(rows):
            texts.extend([
                " | ".join(f"{k}: {str(v)}" for k, v in r.items() if v not in [None, ""])
                for r in chunk if any(r.values())
            ])
        cl.user_session.set("file_table", rows)
        cl.user_session.set("file_headers", headers)

    # ✅ Run FAISS indexing only (no language detection)
    if texts:
        index, st_model = await build_faiss_index(texts)

        cl.user_session.set("file_index", index)
        cl.user_session.set("file_text", "\n".join(texts))
        cl.user_session.set("st_model", st_model)

        return index, texts, st_model

    return None, texts, None

@cl.on_message
async def on_message(msg: cl.Message):
    """
    Balanced unified message handler:
    - Maintains conversation state
    - Handles uploaded files with previews (PDF, DOCX, Excel/CSV, Images)
    - Supports multi-file and single-file queries
    - Routes queries through professional_llm_response for polished output
    """

    # Maintain conversation state
    state = cl.user_session.get("state") or {"messages": []}
    cl.user_session.set("state", state)
    state["messages"].append(HumanMessage(content=msg.content))

    uploaded_files = []
    file_text = ""
    file_path = None
    sidebar_elements = []

    # ✅ Handle uploaded files with unified ingestion + previews
    if msg.elements:
        for element in msg.elements:
            if element.type == "file":
                file_path = element.path
                file_name = element.name
                uploaded_files.append(file_name)
                try:
                    index, texts, st_model = await prepare_file_context(file_path)
                    file_text = "\n".join(texts)
                    cl.user_session.set(f"{file_name}_text", file_text)
                    cl.user_session.set(f"{file_name}_table", cl.user_session.get("file_table"))
                    cl.user_session.set(f"{file_name}_headers", cl.user_session.get("file_headers"))

                    # ✅ Sidebar preview logic
                    if file_path.endswith(".pdf"):
                        sidebar_elements.append(cl.Pdf(path=file_path, name=file_name))
                    elif file_path.endswith(".docx"):
                        preview = "\n".join(texts[:10])
                        sidebar_elements.append(cl.Text(content=preview, name=file_name))
                    elif file_path.endswith((".xlsx", ".csv")):
                        headers = cl.user_session.get("file_headers", [])
                        rows = cl.user_session.get("file_table", [])
                        preview_rows = rows[:5]
                        preview = ""
                        if headers and preview_rows:
                            table = "| " + " | ".join(headers) + " |\n"
                            table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                            for r in preview_rows:
                                row_values = [str(r.get(h, "")) for h in headers]
                                table += "| " + " | ".join(row_values) + " |\n"
                            preview += table
                            stats_summary = compute_all_numeric_stats(rows, headers)
                            preview += "\n\n" + stats_summary
                        else:
                            preview = "\n".join(texts[:10])
                        sidebar_elements.append(cl.Text(content=preview, name=file_name))
                    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        sidebar_elements.append(cl.Image(path=file_path, name=file_name))
                except Exception as e:
                    print(f"⚠️ Error reading file {file_name}: {e}")

    # ✅ Show sidebar previews if any
    if sidebar_elements:
        await cl.ElementSidebar.set_elements(sidebar_elements)
        await cl.ElementSidebar.set_title("Uploaded Files")

    # ✅ Route queries through professional_llm_response
    if len(uploaded_files) > 1:
        # Multi-file query → still use handle_multi_file_query for merging
        answer = await handle_multi_file_query(msg.content, uploaded_files, use_multi_fuzzy=True)
    elif len(uploaded_files) == 1:
        file_table = cl.user_session.get("file_table")
        headers = cl.user_session.get("file_headers", [])
        file_text = cl.user_session.get("file_text", "")
        # 🔹 Always call professional_llm_response for polished narrative
        answer = await professional_llm_response(
            user_query=msg.content,
            file_table=file_table,
            headers=headers,
            llm_runner=cl.make_async(run_qwen_sync)
        )
    else:
        # No file → still let Qwen3 respond professionally
        answer = await professional_llm_response(
            user_query=msg.content,
            file_table=[],
            headers=[],
            llm_runner=cl.make_async(run_qwen_sync)
        )

    await cl.Message(content=answer).send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")

# ------------------- Starters -------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.png"
        ),
        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/learn.png"
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/terminal.png",
            command="code"
        ),
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/write.png"
        )
    ]

# Register tools based on environment flag
register_tools(cl)

