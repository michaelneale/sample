"""Utility functions for document operations."""

import re
from typing import Any


async def execute_request(
    method: str, document_id: str, request: dict, **kwargs
) -> dict[str, Any]:
    """Execute a single document API request."""
    raise NotImplementedError("Stub: would call document backend")


async def execute_requests(
    document_id: str, requests: list[dict], **kwargs
) -> dict[str, Any]:
    """Execute a batch of document API requests."""
    raise NotImplementedError("Stub: would call document backend")


def calculate_utf16_length(text: str) -> int:
    """Calculate text length in UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def hex_to_rgb(hex_color: str) -> dict:
    """Convert hex color to RGB format (0.0-1.0 range)."""
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    if not re.match(r"^[0-9A-Fa-f]{6}$", hex_color):
        raise ValueError(f"Invalid hex color: {hex_color}")

    return {
        "color": {
            "rgbColor": {
                "red": int(hex_color[0:2], 16) / 255.0,
                "green": int(hex_color[2:4], 16) / 255.0,
                "blue": int(hex_color[4:6], 16) / 255.0,
            }
        }
    }


def build_text_style(
    foreground_color: str | None = None,
    background_color: str | None = None,
    font_size: int | None = None,
    font_family: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    underline: bool | None = None,
    strikethrough: bool | None = None,
    link_url: str | None = None,
) -> tuple[dict, str]:
    """Build text style object and field mask from provided formatting options."""
    style = {}
    fields = []

    if foreground_color:
        style["foregroundColor"] = hex_to_rgb(foreground_color)
        fields.append("foregroundColor")

    if background_color:
        style["backgroundColor"] = hex_to_rgb(background_color)
        fields.append("backgroundColor")

    if font_size is not None:
        style["fontSize"] = {"magnitude": font_size, "unit": "PT"}
        fields.append("fontSize")

    if font_family:
        style["weightedFontFamily"] = {"fontFamily": font_family}
        fields.append("weightedFontFamily")

    if bold is not None:
        style["bold"] = bold
        fields.append("bold")

    if italic is not None:
        style["italic"] = italic
        fields.append("italic")

    if underline is not None:
        style["underline"] = underline
        fields.append("underline")

    if strikethrough is not None:
        style["strikethrough"] = strikethrough
        fields.append("strikethrough")

    if link_url:
        style["link"] = {"url": link_url}
        fields.append("link")

    return style, ",".join(fields)


def find_text_positions(
    document: dict, search_text: str
) -> list[tuple[int, int]]:
    """Find all occurrences of text in a document with UTF-16 positions."""
    positions = []
    search_len = calculate_utf16_length(search_text)

    content = document.get("body", {}).get("content", [])
    _find_text_in_elements(content, search_text, search_len, positions)

    return positions


def _find_text_in_elements(
    elements: list[dict], search_text: str, search_len: int, positions: list
):
    """Recursively find text in document elements."""
    for element in elements:
        if "paragraph" in element:
            para_elements = element["paragraph"].get("elements", [])
            for elem in para_elements:
                if "textRun" in elem:
                    text_run = elem["textRun"]
                    content = text_run.get("content", "")
                    start_index = elem.get("startIndex", 0)

                    idx = 0
                    while True:
                        pos = content.find(search_text, idx)
                        if pos == -1:
                            break
                        prefix_len = calculate_utf16_length(content[:pos])
                        match_start = start_index + prefix_len
                        match_end = match_start + search_len
                        positions.append((match_start, match_end))
                        idx = pos + len(search_text)

        elif "table" in element:
            table_rows = element["table"].get("tableRows", [])
            for row in table_rows:
                for cell in row.get("tableCells", []):
                    cell_content = cell.get("content", [])
                    _find_text_in_elements(cell_content, search_text, search_len, positions)
