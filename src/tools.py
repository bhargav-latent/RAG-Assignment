"""Custom tools and backends for RAG agent."""

import base64
import os
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Will be set by agent.py when agent is created
PAPERS_ROOT: Path | None = None


def get_vision_model() -> ChatGoogleGenerativeAI:
    """Initialize the vision model from environment config."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("VISION_MODEL_NAME", "gemini-3-flash-preview"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_tokens=4096,
    )


@tool(response_format="content_and_artifact")
def read_images(
    image_paths: list[str],
    query: str,
    detail: str = "high",
) -> tuple[str, list[dict]]:
    """Analyze images using a vision model.

    Use this tool when you need to understand figures, diagrams,
    charts, or tables that are stored as images in the papers directory.

    Args:
        image_paths: List of paths to image files (supports PNG, JPG, WEBP, GIF)
        query: Question or instruction about the image(s)
        detail: Image detail level - "low" or "high"

    Returns:
        Tuple of (analysis text, list of image artifacts for UI display)
    """
    vision_model = get_vision_model()

    # Build content with text query first
    content = [{"type": "text", "text": query}]
    artifacts = []

    for path in image_paths:
        image_path = Path(path)

        # Resolve relative paths against PAPERS_ROOT
        if not image_path.is_absolute() and PAPERS_ROOT is not None:
            image_path = PAPERS_ROOT / path

        if not image_path.exists():
            return f"Error: Image not found at {path}", []

        # Read and encode image as base64
        try:
            image_bytes = image_path.read_bytes()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            return f"Error reading image {path}: {e}", []

        # Determine MIME type from extension
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_map.get(suffix, "image/png")

        # Add image in OpenAI format for vision model
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": detail,
            },
        })

        # Build artifact for UI display
        artifacts.append({
            "type": "image",
            "path": str(path),
            "mime_type": mime_type,
            "data": f"data:{mime_type};base64,{image_data}",
        })

    # Call vision model
    try:
        message = HumanMessage(content=content)
        response = vision_model.invoke([message])
        analysis = response.content
    except Exception as e:
        return f"Error calling vision model: {e}", []

    return analysis, artifacts


# ============================================================================
# Read-Only Filesystem Backend
# ============================================================================

try:
    from deepagents.backends import FilesystemBackend
    from deepagents.backends.protocol import WriteResult, EditResult

    class ReadOnlyBackend(FilesystemBackend):
        """A filesystem backend that prevents writes and sandboxes to root_dir.

        Uses virtual_mode=True to enforce path scoping under root_dir.
        Default read limit reduced to 500 lines (from 2000) to control context.
        """

        def __init__(self, root_dir: str):
            # virtual_mode=True sandboxes all paths under root_dir
            super().__init__(root_dir=root_dir, virtual_mode=True)

        def read(self, file_path: str, offset: int = 0, limit: int = 500) -> str:
            """Read file with reduced default limit (500 lines instead of 2000)."""
            return super().read(file_path, offset, limit)

        def write(self, file_path: str, content: str) -> WriteResult:
            return WriteResult(error="Write operations are disabled - papers are read-only")

        def edit(
            self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
        ) -> EditResult:
            return EditResult(error="Edit operations are disabled - papers are read-only")

except ImportError:
    # deepagents not installed yet - provide stub
    ReadOnlyBackend = None
