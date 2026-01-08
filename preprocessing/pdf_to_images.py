"""
PDF to Images Converter
Converts PDF pages to high-resolution images for processing with Nemotron-Parse.
"""

import os
import sys
from pathlib import Path

# Try to import fitz (PyMuPDF) first, fall back to pdf2image
try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except ImportError:
    USE_PYMUPDF = False
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: Neither PyMuPDF nor pdf2image is installed.")
        print("Install with: pip install PyMuPDF")
        print("Or: pip install pdf2image (requires poppler)")
        sys.exit(1)


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    image_format: str = "png"
) -> list[str]:
    """
    Convert a PDF file to images.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images
        dpi: Resolution for conversion (default 300 for Nemotron-Parse)
        image_format: Output format (png or jpg)

    Returns:
        List of paths to generated images
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem
    image_paths = []

    if USE_PYMUPDF:
        print(f"Using PyMuPDF to convert: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            # Calculate zoom factor for desired DPI (default PDF is 72 DPI)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            output_path = output_dir / f"{pdf_name}_page_{page_num + 1:04d}.{image_format}"
            pix.save(str(output_path))
            image_paths.append(str(output_path))

            print(f"  Converted page {page_num + 1}/{total_pages}: {output_path.name}")

        doc.close()
    else:
        print(f"Using pdf2image to convert: {pdf_path.name}")
        images = convert_from_path(str(pdf_path), dpi=dpi)
        total_pages = len(images)

        for page_num, image in enumerate(images):
            output_path = output_dir / f"{pdf_name}_page_{page_num + 1:04d}.{image_format}"
            image.save(str(output_path), image_format.upper())
            image_paths.append(str(output_path))

            print(f"  Converted page {page_num + 1}/{total_pages}: {output_path.name}")

    print(f"Successfully converted {len(image_paths)} pages")
    return image_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_images.py <pdf_path> [output_dir] [dpi]")
        print("Example: python pdf_to_images.py ../Science-Datasets.pdf ../data/images 300")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/images"
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    image_paths = convert_pdf_to_images(pdf_path, output_dir, dpi)
    print(f"\nGenerated {len(image_paths)} images in {output_dir}")
