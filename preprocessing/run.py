"""
PDF to Markdown Pipeline
Extracts text and detected images using NVIDIA Nemotron-Parse (bbox mode).
"""

import os
import sys
import asyncio
import aiohttp
import base64
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
load_dotenv(Path(__file__).parent.parent / ".env")

from config import (
    ROOT_DIR, DATA_DIR, IMAGES_DIR, PAPERS_DIR,
    NVAI_URL, MODEL, MAX_PARALLEL, TIMEOUT, DPI
)
from pdf_to_images import convert_pdf_to_images

# Image types to extract
IMAGE_TYPES = {'picture', 'figure', 'image', 'chart', 'diagram'}


def get_api_key():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY not found in .env")
    return key


def read_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def parse_bbox_response(result):
    """Extract text and image bboxes from Nemotron bbox response."""
    try:
        choices = result.get("choices", [])
        if not choices:
            return None, []

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return message.get("content"), []

        args = tool_calls[0].get("function", {}).get("arguments", "")
        if isinstance(args, str):
            args = json.loads(args)

        # Handle nested list structure: args[0] contains the actual items
        items = args
        if isinstance(args, list) and len(args) > 0 and isinstance(args[0], list):
            items = args[0]

        texts = []
        images = []

        for item in items:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "").lower()
            text = item.get("text", "")
            bbox = item.get("bbox", {})

            # Extract images
            if item_type in IMAGE_TYPES:
                if bbox:
                    images.append({
                        "bbox": bbox,
                        "type": item_type,
                        "caption": ""
                    })
            # Captions (associate with previous image)
            elif item_type == "caption" and images:
                images[-1]["caption"] = text
            # Regular text
            elif text:
                texts.append(text)

        return "\n\n".join(texts), images

    except Exception as e:
        print(f"    Parse error: {e}")
        return None, []


def crop_image(page_image_path, bbox, output_path):
    """Crop detected region from page image."""
    try:
        img = Image.open(page_image_path)
        w, h = img.size

        # bbox has xmin, ymin, xmax, ymax (normalized 0-1)
        x1 = int(bbox.get("xmin", 0) * w)
        y1 = int(bbox.get("ymin", 0) * h)
        x2 = int(bbox.get("xmax", 1) * w)
        y2 = int(bbox.get("ymax", 1) * h)

        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(output_path)
        return True
    except Exception as e:
        print(f"    Crop error: {e}")
        return False


async def extract_page(session, image_path, api_key):
    """Extract markdown and image bboxes from a single page."""
    b64 = read_image_base64(image_path)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f'<img src="data:image/png;base64,{b64}" />'}],
        "tools": [{"type": "function", "function": {"name": "markdown_bbox"}}],
        "tool_choice": {"type": "function", "function": {"name": "markdown_bbox"}},
        "max_tokens": 8192,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with session.post(NVAI_URL, headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
            resp.raise_for_status()
            return parse_bbox_response(await resp.json())
    except Exception as e:
        print(f"    API error: {e}")
        return None, []


async def process_batch(session, batch, api_key):
    """Process a batch of pages in parallel."""
    tasks = [extract_page(session, img, api_key) for img, _ in batch]
    return await asyncio.gather(*tasks)


def extract_figures(page_num, detected_images, page_image_path, figures_dir):
    """Extract and save figures from a page (no individual page files)."""
    num_figures = 0

    for i, img_info in enumerate(detected_images):
        bbox = img_info.get("bbox")
        if bbox:
            fig_name = f"page_{page_num:04d}_fig_{i+1:02d}.png"
            fig_path = figures_dir / fig_name
            if crop_image(page_image_path, bbox, fig_path):
                num_figures += 1

    return num_figures


def save_page_markdown(page_num, markdown, detected_images, page_image_path,
                       paper_dir, figures_dir):
    """Legacy function - extracts figures only (no page files created)."""
    num_figures = extract_figures(page_num, detected_images, page_image_path, figures_dir)
    return None, num_figures


def save_combined_markdown(pages_data, paper_dir, figures_dir, pdf_name):
    """Save all pages into a single markdown file."""
    combined_path = paper_dir / "document.md"

    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_name}\n\n")

        for page_num, markdown, detected_images, _ in pages_data:
            f.write(f"## Page {page_num}\n\n")

            # Add detected figures
            for i, img_info in enumerate(detected_images):
                fig_name = f"page_{page_num:04d}_fig_{i+1:02d}.png"
                fig_path = figures_dir / fig_name

                if fig_path.exists():
                    rel_path = os.path.relpath(fig_path, paper_dir)
                    img_type = img_info.get("type", "figure")
                    caption = img_info.get("caption", "")
                    f.write(f"![{img_type} {i+1}]({rel_path})\n")
                    if caption:
                        f.write(f"*{caption}*\n")
                    f.write("\n")

            if markdown:
                f.write(f"{markdown}\n\n")

            f.write("---\n\n")

    return combined_path


async def run_pipeline(pdf_path):
    """Main pipeline: PDF -> Images -> Markdown with detected figures."""
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    # Setup directories
    images_dir = IMAGES_DIR / pdf_name  # Temp page images
    paper_dir = PAPERS_DIR / pdf_name   # Output for agent
    figures_dir = paper_dir / "figures" # Extracted figures

    images_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PDF: {pdf_path.name}")
    print(f"Mode: markdown_bbox (extracts detected figures)")
    print(f"{'='*60}")

    # Step 1: Convert PDF to images
    print("\n[1/2] Converting PDF to images...")
    start = time.time()
    image_paths = convert_pdf_to_images(str(pdf_path), str(images_dir), dpi=DPI)
    print(f"      {len(image_paths)} pages in {time.time()-start:.1f}s")

    # Step 2: Extract markdown + detected images
    print(f"\n[2/2] Extracting text + figures ({MAX_PARALLEL} parallel)...")
    start = time.time()

    api_key = get_api_key()
    all_results = []
    total_figures = 0

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(image_paths), MAX_PARALLEL):
            batch = [(img, idx+1) for idx, img in enumerate(image_paths[i:i+MAX_PARALLEL], start=i)]
            batch_num = i // MAX_PARALLEL + 1
            total_batches = (len(image_paths) + MAX_PARALLEL - 1) // MAX_PARALLEL

            print(f"      Batch {batch_num}/{total_batches}...", end=" ", flush=True)

            results = await process_batch(session, batch, api_key)

            # Save individual pages
            batch_figs = 0
            for (img_path, page_num), (markdown, detected_images) in zip(batch, results):
                _, num_figs = save_page_markdown(
                    page_num, markdown, detected_images, img_path,
                    paper_dir, figures_dir
                )
                all_results.append((page_num, markdown, detected_images, img_path))
                batch_figs += num_figs
                total_figures += num_figs

            success = sum(1 for md, _ in results if md)
            print(f"{success}/{len(results)} OK, {batch_figs} figures")

    # Save combined file
    combined = save_combined_markdown(all_results, paper_dir, figures_dir, pdf_name)

    # Summary
    elapsed = time.time() - start
    successful = sum(1 for _, md, _, _ in all_results if md)
    total_chars = sum(len(md) for _, md, _, _ in all_results if md)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Pages:      {successful}/{len(image_paths)}")
    print(f"Figures:    {total_figures} extracted")
    print(f"Characters: {total_chars:,}")
    print(f"Time:       {elapsed:.1f}s ({len(image_paths)/elapsed:.2f} pages/sec)")
    print(f"\nOutput:")
    print(f"  Figures:  {figures_dir}")
    print(f"  Markdown: {paper_dir}")
    print(f"  Combined: {combined}")

    return all_results


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else ROOT_DIR / "Science-Datasets.pdf"
    asyncio.run(run_pipeline(pdf))
