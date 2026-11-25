"""Deterministic reading of retrieved rows, shown when the LLM call fails.

The user then still sees the facts from the knowledge graph, without any language model.
"""

NO_DATA = "The knowledge graph has no data for that question."
MAX_LISTED = 10


def _cell(value) -> str:
    """Show whole floats as integers (12.0 -> 12) and everything else as text."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _row_text(row: dict) -> str:
    """One row as 'column: value, column: value', skipping missing values and id columns."""
    return ", ".join(f"{key}: {_cell(value)}" for key, value in row.items()
                     if value is not None and not key.lower().endswith("id"))


def read_rows(intent: str, retrieval) -> str:
    """Plain-language reading of `retrieval.rows`: one sentence, a numbered list, or 'no data'."""
    rows = retrieval.rows
    if not rows:
        return NO_DATA
    # A single row reads as one sentence.
    if len(rows) == 1:
        return f"Result: {_row_text(rows[0])}. {retrieval.scope}"
    # Several rows become a short numbered list.
    lines = [f"{i}. {_row_text(row)}" for i, row in enumerate(rows[:MAX_LISTED], 1)]
    if len(rows) > MAX_LISTED:
        lines.append(f"... and {len(rows) - MAX_LISTED} more rows.")
    return "Results:\n" + "\n".join(lines) + f"\n{retrieval.scope}"
