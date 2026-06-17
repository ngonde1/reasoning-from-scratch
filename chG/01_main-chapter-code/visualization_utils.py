from schema_utils import compute_checkout, compute_profit, resolve_header
from dateutil.parser import parse as _parse_date
import plotly.graph_objects as go
import plotly.express as px   # 🔹 added for Gantt chart
import chainlit as cl
import pandas as pd
import plotly.figure_factory as ff

# ------------------ Chart Builders ------------------

def build_bar_chart(rows, column, title=None):
    values, labels = [], []
    for row in rows:
        val = row.get(column)
        if isinstance(val, (int, float)):
            values.append(val)
            labels.append(str(row.get("Task Name", "?")))
    fig = go.Figure(
        data=[go.Bar(
            x=labels,
            y=values,
            marker=dict(color=values, colorscale="RdYlGn", cmin=0, cmax=1)
        )]
    )
    fig.update_layout(title=title or f"{column} Bar Chart")
    return cl.Plotly(name=title or column, figure=fig, display="side")


def build_pie_chart(rows, column, title=None):
    labels = [str(row.get(column, "?")) for row in rows if row.get(column)]
    values = [1 for _ in labels]  # count each occurrence
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title=title or f"{column} Pie Chart")
    return cl.Plotly(name=title or column, figure=fig, display="side")


def build_line_chart(rows, column, title=None):
    values, labels = [], []
    for i, row in enumerate(rows):
        val = row.get(column)
        if isinstance(val, (int, float)):
            values.append(val)
            labels.append(i)
    fig = go.Figure(data=[go.Scatter(x=labels, y=values, mode="lines+markers")])
    fig.update_layout(title=title or f"{column} Line Chart")
    return cl.Plotly(name=title or column, figure=fig, display="side")


def build_count_chart(rows, column, title=None):
    counts = {}
    for row in rows:
        val = row.get(column)
        if val:
            counts[val] = counts.get(val, 0) + 1
    labels, values = list(counts.keys()), list(counts.values())
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="blue")])
    fig.update_layout(title=title or f"{column} Count Chart")
    return cl.Plotly(name=title or column, figure=fig, display="side")


def build_gantt_chart(rows, title="Timeline Chart"):
    """
    Universal Gantt chart builder.
    Auto-detects start/end columns, labels, grouping, and color coding.
    """
    if not rows:
        return cl.Text(content="No data available.")

    headers = rows[0].keys()
    start_col = next((h for h in headers if "start" in h.lower()), None)
    end_col = next((h for h in headers if "end" in h.lower() or "finish" in h.lower()), None)
    label_col = next((h for h in headers if any(x in h.lower() for x in ["task","activity","event","name"])), None)
    group_col = next((h for h in headers if any(x in h.lower() for x in ["project","category","department"])), None)
    color_col = next((h for h in headers if any(x in h.lower() for x in ["owner","assigned","person","user"])), None)

    if not start_col or not end_col:
        return cl.Text(content="No start/end columns detected for timeline.")

    df = pd.DataFrame(rows)

    if not label_col:
        df["Row"] = range(len(df))
        label_col = "Row"

    fig = px.timeline(
        df,
        x_start=start_col,
        x_end=end_col,
        y=label_col,
        color=color_col if color_col else label_col,
        facet_row=group_col if group_col else None,
        title=title
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, showlegend=True)

    return cl.Plotly(name=title, figure=fig, display="side")

# ------------------ Dispatcher ------------------
def build_chart_auto(rows, column, user_prompt, title=None):
    q = user_prompt.lower()

    # --- Derived field detection ---
    if "profit" in q:
        revenue_col = resolve_header(list(rows[0].keys()), "revenue")
        expense_col = resolve_header(list(rows[0].keys()), "expense")
        enriched = []
        for row in rows:
            profit = compute_profit(row, revenue_col, expense_col)
            if profit is not None:
                enriched.append({"Profit": profit, **row})
        if not enriched:
            return cl.Text(content="No profit values found.")
        fig = px.bar(enriched, x=list(rows[0].keys())[0], y="Profit", title=title or "Profit Chart")
        return [cl.Plotly(name="Profit", figure=fig, display="side")]

    if "checkout" in q or "timeline" in q:
        start_col = resolve_header(list(rows[0].keys()), "start")
        nights_col = resolve_header(list(rows[0].keys()), "progress") or "Nights"
        enriched = []
        for row in rows:
            checkout = compute_checkout(row, start_col, nights_col)
            if checkout:
                enriched.append({
                    "Task": row.get(resolve_header(list(rows[0].keys()), "name"), "Unknown"),
                    "Start": row.get(start_col),
                    "Finish": checkout
                })
        if not enriched:
            return cl.Text(content="No checkout dates found.")
        fig = px.timeline(enriched, x_start="Start", x_end="Finish", y="Task", title=title or "Checkout Timeline")
        fig.update_yaxes(autorange="reversed")
        return [cl.Plotly(name="Checkout Timeline", figure=fig, display="side")]

    # --- Explicit user intent detection ---
    if "timeline" in q or "gantt" in q or ("start" in q and "end" in q) or "schedule" in q:
        return [build_gantt_chart(rows, title or "Timeline Chart")]
    if "pie" in q:
        return [build_pie_chart(rows, column, title)]
    if "line" in q or "trend" in q:
        return [build_line_chart(rows, column, title)]
    if "count" in q or "tasks per" in q:
        return [build_count_chart(rows, column, title)]
    if "bar" in q or "graph" in q or "plot" in q:
        return [build_bar_chart(rows, column, title)]

    # --- Auto-detection based on column data ---
    sample_val = next((row.get(column) for row in rows if row.get(column) is not None), None)

    # Detect timeline if start/end columns exist
    headers = rows[0].keys() if rows else []
    has_start = any("start" in h.lower() for h in headers)
    has_end = any("end" in h.lower() or "finish" in h.lower() for h in headers)
    if has_start and has_end:
        return [build_gantt_chart(rows, title or "Timeline Chart")]

    charts = []

    # Numeric → suggest both line and bar charts
    if isinstance(sample_val, (int, float)):
        charts.append(build_bar_chart(rows, column, title or f"{column} Bar Chart"))
        charts.append(build_line_chart(rows, column, title or f"{column} Trend"))
        return charts

    # Categorical → suggest both pie and count charts
    if isinstance(sample_val, str):
        unique_vals = {row.get(column) for row in rows if row.get(column)}
        if len(unique_vals) > 5:
            charts.append(build_count_chart(rows, column, title or f"{column} Count Chart"))
        else:
            charts.append(build_pie_chart(rows, column, title or f"{column} Pie Chart"))
        # Always add count chart as a second option
        charts.append(build_count_chart(rows, column, title or f"{column} Count Chart"))
        return charts

    # Fallback
    return [build_count_chart(rows, column, title or f"{column} Count Chart")]

# ------------------ Multi-file Handler ------------------

def build_charts_for_files(files_dict, column, user_prompt):
    """
    Build charts for the same column across multiple uploaded files.
    Supports derived fields (Profit, Checkout) aggregated across all files.
    files_dict: {file_name: rows}
    Returns a list of cl.Plotly charts (can include multiple per file or aggregated).
    """
    charts = []
    q = user_prompt.lower()

    # --- Derived: Profit across all files ---
    if "profit" in q:
        combined = []
        for fname, rows in files_dict.items():
            revenue_col = resolve_header(list(rows[0].keys()), "revenue")
            expense_col = resolve_header(list(rows[0].keys()), "expense")
            for row in rows:
                profit = compute_profit(row, revenue_col, expense_col)
                if profit is not None:
                    combined.append({"File": fname, "Profit": profit})
        if combined:
            fig = px.bar(combined, x="File", y="Profit", title="Combined Profit Across Files")
            charts.append(cl.Plotly(name="Combined Profit", figure=fig, display="side"))

    # --- Derived: Checkout timeline across all files ---
    if "checkout" in q or "timeline" in q:
        combined = []
        for fname, rows in files_dict.items():
            start_col = resolve_header(list(rows[0].keys()), "start")
            nights_col = resolve_header(list(rows[0].keys()), "progress") or "Nights"
            for row in rows:
                checkout = compute_checkout(row, start_col, nights_col)
                if checkout:
                    combined.append({
                        "Task": row.get(resolve_header(list(rows[0].keys()), "name"), "Unknown"),
                        "Start": row.get(start_col),
                        "Finish": checkout,
                        "File": fname
                    })
        if combined:
            fig = ff.create_gantt(combined, index_col="File", show_colorbar=True,
                                  title="Combined Checkout Timeline")
            charts.append(cl.Plotly(name="Combined Checkout Timeline", figure=fig, display="side"))

    # --- Standard charts per file (fallback) ---
    for file_name, rows in files_dict.items():
        file_charts = build_chart_auto(rows, column, user_prompt, title=f"{file_name} – {column}")
        if isinstance(file_charts, list):
            charts.extend(file_charts)
        else:
            charts.append(file_charts)

    return charts