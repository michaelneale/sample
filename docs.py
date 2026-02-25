"""Tools for interacting with documents."""

import json
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal

from pydantic import Field

from doc_utils import execute_request, execute_requests

DOC_OPERATIONS = Literal[
    "get_document",
    "insert_text",
    "append_text",
    "replace_text",
    "delete_content",
    "insert_table",
    "update_table_cell",
    "insert_table_row",
    "insert_table_column",
    "delete_table_row",
    "delete_table_column",
    "insert_image",
    "format_existing_text",
]

RESPONSE_CHAR_LIMIT = 400000


@dataclass
class FormatTextParams:
    """Parameters for format_existing_text operation."""

    search_text: str
    foreground_color: str | None = None
    background_color: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    strikethrough: bool | None = None
    font_size: int | None = None
    font_family: str | None = None
    heading_level: int | None = None
    link_url: str | None = None
    list_type: str | None = None


@dataclass
class TableParams:
    """Parameters for table operations.

    Used by operations: insert_table, update_table_cell, insert_table_row,
    insert_table_column, delete_table_row, delete_table_column.
    """

    rows: int | None = Field(None, description="Number of rows for insert_table operation")
    columns: int | None = Field(None, description="Number of columns for insert_table operation")
    row_index: int | None = Field(
        None, description="Row index (0-based) for update_table_cell, insert_table_row, and delete_table_row"
    )
    column_index: int | None = Field(
        None, description="Column index (0-based) for update_table_cell, insert_table_column, and delete_table_column"
    )
    insert_below: bool = Field(
        False, description="For insert_table_row: True to insert below the specified row, False to insert above"
    )
    insert_right: bool = Field(
        False, description="For insert_table_column: True to insert right of column, False to insert left"
    )


@dataclass
class ImageParams:
    """Parameters for image insertion."""

    image_url: str = Field(..., description="URL of the image to insert (must be publicly accessible)")
    width: int | None = Field(None, description="Width of the image in points (PT)")
    height: int | None = Field(None, description="Height of the image in points (PT)")


async def doc_tool(
    document_id: str,
    operation: DOC_OPERATIONS = "get_document",
    text: Annotated[
        str,
        Field(
            description=(
                "Text content to insert, append, or match for replacement. "
                "For insert_text and append_text, Markdown formatting is supported. "
                "For replace_text, this should be unformatted plain text."
            )
        ),
    ] = "",
    replace_text: Annotated[
        str,
        Field(
            description=(
                "New plain text that will replace all occurrences of the original text. "
                "Only used for the replace_text operation."
            )
        ),
    ] = "",
    start_position: Annotated[
        int | None,
        Field(
            description="Document index (1-based) for insert or delete operations.",
            ge=1,
        ),
    ] = None,
    end_position: Annotated[
        int | None,
        Field(
            description="Document index (1-based, exclusive) for delete_content",
            ge=1,
        ),
    ] = None,
    table_params: Annotated[
        TableParams | None,
        Field(
            None,
            description="Parameters for table operations",
        ),
    ] = None,
    image_params: ImageParams | None = None,
    format_params: FormatTextParams | None = None,
) -> str:
    """Perform operations on an existing document.

    Supported operations:
        - get_document: Returns document content
        - insert_text: Inserts text at a specific position
        - append_text: Appends text at the end of the document
        - replace_text: Replaces all instances of text with replace_text
        - delete_content: Deletes content between two positions
        - insert_table: Creates a table with specified rows and columns
        - update_table_cell: Updates content in a specific table cell
        - insert_table_row: Inserts a row above or below the specified row
        - insert_table_column: Inserts a column left or right of the specified column
        - delete_table_row: Deletes the specified row from a table
        - delete_table_column: Deletes the specified column from a table
        - insert_image: Inserts an image from a URL at the specified position
        - format_existing_text: Finds and applies formatting to text
    """
    if not text and operation in ["insert_text", "append_text", "replace_text"]:
        raise ValueError(f"text is required for {operation} operation")

    table_operations = [
        "insert_table",
        "update_table_cell",
        "insert_table_row",
        "insert_table_column",
        "delete_table_row",
        "delete_table_column",
    ]
    if operation in table_operations and not table_params:
        raise ValueError(f"table_params is required for {operation} operation")

    if operation == "insert_image" and not image_params:
        raise ValueError("image_params is required for insert_image operation")

    if operation == "format_existing_text" and not format_params:
        raise ValueError("format_params is required for format_existing_text operation")

    operation_handlers = {
        "get_document": partial(read_document, document_id),
        "insert_text": partial(insert_text, document_id, text, start_position),
        "append_text": partial(append_text, document_id, text),
        "replace_text": partial(replace_all_text, document_id, text, replace_text),
        "delete_content": partial(delete_content, document_id, start_position, end_position),
        "insert_table": partial(
            insert_table,
            document_id,
            table_params.rows if table_params else None,
            table_params.columns if table_params else None,
            start_position,
        ),
        "update_table_cell": partial(
            update_table_cell,
            document_id,
            table_params.row_index if table_params else None,
            table_params.column_index if table_params else None,
            text,
            start_position,
        ),
        "insert_table_row": partial(
            modify_table_structure,
            document_id,
            "insert_row",
            table_params.row_index if table_params else None,
            None,
            table_params.insert_below if table_params else False,
            False,
            start_position,
        ),
        "insert_table_column": partial(
            modify_table_structure,
            document_id,
            "insert_column",
            None,
            table_params.column_index if table_params else None,
            False,
            table_params.insert_right if table_params else False,
            start_position,
        ),
        "delete_table_row": partial(
            modify_table_structure,
            document_id,
            "delete_row",
            table_params.row_index if table_params else None,
            None,
            False,
            False,
            start_position,
        ),
        "delete_table_column": partial(
            modify_table_structure,
            document_id,
            "delete_column",
            None,
            table_params.column_index if table_params else None,
            False,
            False,
            start_position,
        ),
        "insert_image": partial(
            insert_image,
            document_id,
            image_params.image_url if image_params else None,
            start_position,
            image_params.width if image_params else None,
            image_params.height if image_params else None,
        ),
        "format_existing_text": partial(
            format_existing_text,
            document_id,
            format_params.search_text if format_params else None,
            format_params.foreground_color if format_params else None,
            format_params.background_color if format_params else None,
            format_params.font_size if format_params else None,
            format_params.font_family if format_params else None,
            format_params.bold if format_params else None,
            format_params.italic if format_params else None,
            format_params.underline if format_params else None,
            format_params.strikethrough if format_params else None,
            format_params.heading_level if format_params else None,
            format_params.link_url if format_params else None,
            format_params.list_type if format_params else None,
        ),
    }

    if operation not in operation_handlers:
        raise ValueError(f"Invalid operation: {operation}")

    response = await operation_handlers[operation]()
    return json.dumps(response, indent=2)


def _extract_text_from_element(element: dict) -> str:
    """Recursively pull text from paragraphs, tables, etc."""
    text_parts = ""
    if "paragraph" in element:
        paragraph_elements = element["paragraph"].get("elements", [])
        for el in paragraph_elements:
            if "textRun" in el:
                content = el["textRun"]["content"]
                url = el["textRun"].get("textStyle", {}).get("link", {}).get("url")
                if url:
                    text_parts += f"[{content}]({url})"
                else:
                    text_parts += content
    elif "table" in element:
        table_rows = element["table"].get("tableRows", [])
        for row in table_rows:
            for cell in row["tableCells"]:
                for cell_content in cell["content"]:
                    text_parts += _extract_text_from_element(cell_content)
        text_parts += "\n"
    return text_parts


async def read_document(document_id: str) -> dict[str, Any]:
    """Returns document content."""
    document = await execute_request("get", document_id, {})
    content = document.get("body", {}).get("content", [])
    text = "".join(_extract_text_from_element(e) for e in content)

    result = {"content": text, "document_id": document_id}

    if len(json.dumps(result)) > RESPONSE_CHAR_LIMIT:
        raise ValueError(f"Document {document_id} is too large to read.")

    return result


async def insert_text(
    document_id: str, text: str, start_position: int | None
) -> dict[str, Any]:
    """Insert text at a specific index in a document."""
    if start_position is None:
        raise ValueError("start_position is required for insert_text operation")

    request = {"insertText": {"location": {"index": start_position}, "text": text}}
    await execute_request("update", document_id, request)

    from doc_utils import calculate_utf16_length

    inserted_length = calculate_utf16_length(text)

    return {
        "message": f"Inserted text at position {start_position}",
        "text": text,
        "start_position": start_position,
        "end_position": start_position + inserted_length,
    }


async def append_text(document_id: str, text: str) -> dict[str, Any]:
    """Append text to the end of a document."""
    end_index = await get_document_last_index(document_id)
    if text[0] != "\n":
        text = "\n" + text
    response = await insert_text(document_id, text, end_index - 1)
    response["message"] = "Appended text to the end of the document"
    return response


async def replace_all_text(
    document_id: str, text: str, replace_text: str
) -> dict[str, str]:
    """Replace all instances of a string in a document."""
    if not replace_text:
        raise ValueError("replace_text parameter is required for replace_text operation")

    request = {
        "replaceAllText": {
            "containsText": {"text": text, "matchCase": True},
            "replaceText": replace_text,
        }
    }

    response = await execute_request("update", document_id, request)
    occurrences = response.get("occurrencesChanged", 0)
    if occurrences == 0:
        return {
            "message": f"No occurrences of text '{text}' found in the document.",
            "text": text,
            "replace_text": replace_text,
        }

    return {
        "message": f"Replaced {occurrences} occurrences of '{text}' with '{replace_text}'",
        "text": text,
        "replace_text": replace_text,
    }


async def delete_content(
    document_id: str, start_index: int | None, end_index: int | None
) -> dict[str, Any]:
    """Delete content between two positions."""
    if start_index is None or end_index is None:
        raise ValueError("Both start_index and end_index are required for delete_content")

    request = {
        "deleteContentRange": {
            "range": {"startIndex": start_index, "endIndex": end_index}
        }
    }
    await execute_request("update", document_id, request)
    return {
        "message": f"Deleted content between index {start_index} and {end_index}",
        "start_index": start_index,
        "end_index": end_index,
    }


async def get_document_last_index(document_id: str) -> int:
    """Get the last index of the document."""
    document = await execute_request("get", document_id, {})
    return document.get("body", {}).get("content", [])[-1].get("endIndex", 1)


async def insert_table(
    document_id: str, rows: int | None, columns: int | None, start_position: int | None
) -> str:
    """Insert a table at a specific position."""
    if not rows or not columns:
        raise ValueError("rows and columns are required for insert_table operation")
    if start_position is None:
        raise ValueError("start_position is required for insert_table operation")

    request = {
        "insertTable": {
            "rows": rows,
            "columns": columns,
            "location": {"index": start_position},
        }
    }
    await execute_request("update", document_id, request)
    return f"Inserted {rows}x{columns} table at position {start_position}"


def _table_matches_position(element: dict, start_position: int | None) -> bool:
    """Check if a table element contains the specified position."""
    if start_position is None:
        return True
    table_start = element.get("startIndex")
    table_end = element.get("endIndex")
    return table_start <= start_position < table_end


def _find_table_cell_range(
    document: dict, row_index: int, column_index: int, start_position: int | None = None
) -> tuple[int, int] | None:
    """Find the content range within a table cell."""
    content = document.get("body", {}).get("content", [])
    for element in content:
        if "table" in element:
            if not _table_matches_position(element, start_position):
                continue
            table = element["table"]
            if row_index < len(table.get("tableRows", [])):
                row = table["tableRows"][row_index]
                if column_index < len(row.get("tableCells", [])):
                    cell = row["tableCells"][column_index]
                    cell_content = cell.get("content", [])
                    for cell_element in cell_content:
                        if "paragraph" in cell_element:
                            para_start = cell_element.get("startIndex")
                            para_end = cell_element.get("endIndex")
                            return (para_start, para_end)
                    cell_start = cell.get("startIndex")
                    cell_end = cell.get("endIndex")
                    if cell_start is not None and cell_end is not None:
                        return (cell_start + 1, cell_end - 1)
    return None


def _find_table_start_index(
    document: dict, row_index: int, column_index: int, start_position: int | None = None
) -> int | None:
    """Find the start index of a table element."""
    content = document.get("body", {}).get("content", [])
    for element in content:
        if "table" in element:
            if not _table_matches_position(element, start_position):
                continue
            table = element["table"]
            if row_index < len(table.get("tableRows", [])):
                row = table["tableRows"][row_index]
                if column_index < len(row.get("tableCells", [])):
                    return element.get("startIndex")
    return None


async def update_table_cell(
    document_id: str,
    row_index: int | None,
    column_index: int | None,
    text: str,
    start_position: int | None = None,
) -> str:
    """Update content in a specific table cell."""
    if row_index is None or column_index is None:
        raise ValueError("row_index and column_index are required for update_table_cell")

    document = await execute_request("get", document_id, {})
    cell_range = _find_table_cell_range(document, row_index, column_index, start_position)
    if not cell_range:
        raise ValueError(f"Table cell at row {row_index}, col {column_index} not found")

    cell_start, cell_end = cell_range

    requests = []
    if cell_end > cell_start + 1:
        requests.append(
            {"deleteContentRange": {"range": {"startIndex": cell_start, "endIndex": cell_end - 1}}}
        )

    requests.append({"insertText": {"location": {"index": cell_start}, "text": text}})

    await execute_requests(document_id, requests)
    return f"Updated cell at row {row_index}, column {column_index}"


async def modify_table_structure(
    document_id: str,
    operation: str,
    row_index: int | None = None,
    column_index: int | None = None,
    insert_below: bool = False,
    insert_right: bool = False,
    start_position: int | None = None,
) -> str:
    """Modify table structure by inserting or deleting rows/columns."""
    if operation in ["insert_row", "delete_row"] and row_index is None:
        raise ValueError(f"row_index is required for {operation} operation")
    if operation in ["insert_column", "delete_column"] and column_index is None:
        raise ValueError(f"column_index is required for {operation} operation")

    document = await execute_request("get", document_id, {})

    if operation in ["insert_row", "delete_row"]:
        table_start = _find_table_start_index(document, row_index, 0, start_position)
        if not table_start:
            raise ValueError(f"Table row {row_index} not found")
        location = {"tableStartLocation": {"index": table_start}, "rowIndex": row_index}
    else:
        table_start = _find_table_start_index(document, 0, column_index, start_position)
        if not table_start:
            raise ValueError(f"Table column {column_index} not found")
        location = {"tableStartLocation": {"index": table_start}, "columnIndex": column_index}

    operation_map = {
        "insert_row": "insertTableRow",
        "insert_column": "insertTableColumn",
        "delete_row": "deleteTableRow",
        "delete_column": "deleteTableColumn",
    }
    request_key = operation_map[operation]

    if operation == "insert_row":
        request = {request_key: {"tableCellLocation": location, "insertBelow": insert_below}}
        message = f"Inserted row {'below' if insert_below else 'above'} row {row_index}"
    elif operation == "insert_column":
        request = {request_key: {"tableCellLocation": location, "insertRight": insert_right}}
        message = f"Inserted column {'right of' if insert_right else 'left of'} column {column_index}"
    elif operation == "delete_row":
        request = {request_key: {"tableCellLocation": location}}
        message = f"Deleted row {row_index}"
    else:
        request = {request_key: {"tableCellLocation": location}}
        message = f"Deleted column {column_index}"

    await execute_request("update", document_id, request)
    return message


async def insert_image(
    document_id: str,
    image_url: str | None,
    start_position: int | None,
    width: int | None,
    height: int | None,
) -> str:
    """Insert an image from a URL at the specified position."""
    if not image_url:
        raise ValueError("image_url is required for insert_image operation")
    if start_position is None:
        raise ValueError("start_position is required for insert_image operation")

    request = {
        "insertInlineImage": {
            "uri": image_url,
            "location": {"index": start_position},
        }
    }

    if width or height:
        object_size = {}
        if width:
            object_size["width"] = {"magnitude": width, "unit": "PT"}
        if height:
            object_size["height"] = {"magnitude": height, "unit": "PT"}
        request["insertInlineImage"]["objectSize"] = object_size

    await execute_request("update", document_id, request)
    return f"Inserted image from {image_url} at position {start_position}"


async def format_existing_text(
    document_id: str,
    search_text: str | None,
    foreground_color: str | None,
    background_color: str | None,
    font_size: int | None,
    font_family: str | None,
    bold: bool | None,
    italic: bool | None,
    underline: bool | None,
    strikethrough: bool | None,
    heading_level: int | None,
    link_url: str | None,
    list_type: str | None,
) -> str:
    """Find and apply formatting to all occurrences of text."""
    if not search_text:
        raise ValueError("search_text is required for format_existing_text operation")

    document = await execute_request("get", document_id, {})

    from doc_utils import build_text_style, find_text_positions

    positions = find_text_positions(document, search_text)

    if not positions:
        return f"No occurrences of '{search_text}' found"

    text_style, fields = build_text_style(
        foreground_color=foreground_color,
        background_color=background_color,
        font_size=font_size,
        font_family=font_family,
        bold=bold,
        italic=italic,
        underline=underline,
        strikethrough=strikethrough,
        link_url=link_url,
    )

    if not fields and not heading_level and not list_type:
        raise ValueError("At least one formatting option must be specified")

    positions.sort(reverse=True)

    requests = []
    for start, end in positions:
        range_dict = {"startIndex": start, "endIndex": end}

        if fields:
            requests.append(
                {"updateTextStyle": {"range": range_dict, "textStyle": text_style, "fields": fields}}
            )

        if heading_level:
            if not (1 <= heading_level <= 6):
                raise ValueError("heading_level must be between 1 and 6")
            requests.append(
                {
                    "updateParagraphStyle": {
                        "range": range_dict,
                        "paragraphStyle": {"namedStyleType": f"HEADING_{heading_level}"},
                        "fields": "namedStyleType",
                    }
                }
            )

        if list_type:
            if list_type == "bullet":
                preset = "BULLET_DISC_CIRCLE_SQUARE"
            elif list_type == "numbered":
                preset = "NUMBERED_DECIMAL_ALPHA_ROMAN"
            else:
                raise ValueError("list_type must be 'bullet' or 'numbered'")
            requests.append({"createParagraphBullets": {"range": range_dict, "bulletPreset": preset}})

    if not requests:
        raise ValueError(f"No formatting requests generated for '{search_text}'.")

    await execute_requests(document_id, requests)
    return f"Formatted {len(positions)} occurrences of '{search_text}'"
