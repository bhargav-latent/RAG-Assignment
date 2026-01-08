# RAG Assignment

PDF preprocessing pipeline using NVIDIA Nemotron-Parse for text and figure extraction.

## Features

- PDF to image conversion (300 DPI)
- Text extraction with NVIDIA Nemotron-Parse API
- Automatic figure detection and cropping via bounding boxes
- Async processing with parallel API calls (5x speedup)
- Organized markdown output with embedded figure references

## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt

# Configure API key
echo "NVIDIA_API_KEY=your_key_here" > .env
```

## Usage

```bash
cd preprocessing
python run.py ../your_document.pdf
```

## Output Structure

```
data/
├── images/{pdf_name}/
│   ├── {pdf_name}_page_0001.png  # Page images
│   └── figures/                   # Extracted figures
└── markdown/{pdf_name}/
    ├── page_0001.md              # Per-page markdown
    └── {pdf_name}_combined.md    # Combined document
```

## Tech Stack

- **NVIDIA Nemotron-Parse**: Document parsing and OCR
- **PyMuPDF**: PDF to image conversion
- **aiohttp**: Async HTTP requests
- **Pillow**: Image cropping
