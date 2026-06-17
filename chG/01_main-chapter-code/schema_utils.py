# schema_utils.py
import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta

# --- Clean headers ---
def clean_headers(headers):
    cleaned = []
    for h in headers:
        if not h:
            continue
        h = str(h).strip()
        if h.lower().startswith("column"):
            continue
        cleaned.append(h)
    return cleaned

# --- Universal synonyms across domains and languages ---
SYNONYMS = {
    "department": [
        "project name","department","dept","division","team","group","category",
        "département","equipo","abteilung","部门"
    ],
    "start": [
        "start date","start","begin","from","checkin","arrival","join date","hire date",
        "fecha inicio","date début","anfang","开始"
    ],
    "end": [
        "end date","end","finish","to","checkout","departure","leave date","termination date",
        "fecha fin","date fin","ende","结束"
    ],
    "progress": [
        "progress","completion","percent","pct","status","done","completion %","nights",
        "staylength","duration","score","grade","performance","revenue","amount","salary",
        "cost","expense","avance","progreso","fortschritt","进度"
    ],
    "name": [
        "assigned to","task name","employee","assignee","owner","person","guestname",
        "customer","client","staff","worker","user","member","student","vendor","supplier",
        "nom","nombre","name","姓名"
    ]
}

def _normalize(h: str) -> str:
    return (h or "").strip().lower().replace(" ", "")

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

# --- Resolve header with synonyms + fuzzy matching ---
def resolve_header(headers, target, threshold=0.6):
    if not headers:
        return None
    headers = clean_headers(headers)

    # Synonym match
    for h in headers:
        for syn in SYNONYMS.get(target, []):
            if syn.replace(" ", "").lower() in _normalize(h):
                return h

    # Fuzzy similarity
    best_h, best_score = None, 0.0
    for h in headers:
        for syn in SYNONYMS.get(target, []):
            score = _similarity(h, syn)
            if score > best_score:
                best_h, best_score = h, score
    if best_score >= threshold:
        return best_h

    return None

# --- Derived field computations ---
def compute_checkout(row, start_col, nights_col):
    """Compute checkout date from checkin + nights."""
    try:
        checkin = row.get(start_col)
        nights = row.get(nights_col)
        if isinstance(checkin, datetime) and isinstance(nights, (int, float)):
            return checkin + timedelta(days=int(nights))
        if isinstance(checkin, str) and isinstance(nights, (int, float)):
            dt = datetime.fromisoformat(checkin)
            return dt + timedelta(days=int(nights))
    except Exception:
        return None
    return None

def compute_profit(row, revenue_col, expense_col):
    """Compute profit from revenue - expense."""
    try:
        revenue = row.get(revenue_col, 0)
        expense = row.get(expense_col, 0)
        if isinstance(revenue, (int, float)) and isinstance(expense, (int, float)):
            return revenue - expense
    except Exception:
        return None
    return None
