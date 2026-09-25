"""Turn an uploaded file into page images the vision model can read."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from PIL import Image

MAX_PAGES = 3          # guardrail: invoices are short; stops huge PDFs from burning tokens
MAX_FILE_MB = 10
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
# Vision models bill by image size. 1400 px on the long side keeps printed text readable
# while using far fewer tokens than a full-resolution scan.
MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1400"))


def _shrink(img: Image.Image) -> tuple[bytes, str]:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


class DocumentError(ValueError):
    """The file can't be processed (wrong type, too big, unreadable)."""


def load_pages(path: str | Path) -> list[tuple[bytes, str]]:
    """Return [(image_bytes, mime_type), ...], one entry per page."""
    path = Path(path)
    if not path.exists():
        raise DocumentError(f"File not found: {path}")
    if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        raise DocumentError(f"File is larger than {MAX_FILE_MB} MB")

    suffix = path.suffix.lower()
    if suffix in IMAGE_TYPES:
        try:
            with Image.open(path) as img:
                return [_shrink(img)]
        except OSError as exc:
            raise DocumentError(f"Unreadable image: {exc}") from exc
    if suffix == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            if doc.page_count > MAX_PAGES:
                raise DocumentError(f"PDF has {doc.page_count} pages; the limit is {MAX_PAGES}")
            for page in doc:
                pix = page.get_pixmap(dpi=130)
                pages.append(_shrink(Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
        return pages
    raise DocumentError(f"Unsupported file type: {suffix}")