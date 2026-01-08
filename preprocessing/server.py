"""FastAPI server for PDF upload and preprocessing."""

import asyncio
import shutil
import uuid
import time
import aiohttp
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from config import ROOT_DIR, RAW_PDFS_DIR, PAPERS_DIR, IMAGES_DIR, NVAI_URL, MODEL, MAX_PARALLEL, TIMEOUT, DPI
from pdf_to_images import convert_pdf_to_images
from run import (
    get_api_key, extract_page, save_page_markdown, save_combined_markdown
)

app = FastAPI(title="PDF Preprocessor", description="Upload PDFs for RAG preprocessing")

# Ensure directories exist
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Track processing jobs
jobs: dict[str, dict] = {}


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Preprocessor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
        }
        .container { max-width: 800px; margin: 0 auto; position: relative; z-index: 10; }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .subtitle { color: #888; margin-bottom: 2rem; }

        /* Floating particles */
        .particles {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            z-index: 1;
            opacity: 0;
            transition: opacity 0.5s;
        }
        .particles.active { opacity: 1; }
        .particle {
            position: absolute;
            font-size: 2.5rem;
            opacity: 0.15;
            animation: float 20s infinite linear;
        }
        @keyframes float {
            0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            10% { opacity: 0.2; }
            90% { opacity: 0.2; }
            100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
        }

        .upload-zone {
            border: 2px dashed #333;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: #111;
            position: relative;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: #0ea5e9;
            background: #0c1929;
        }
        .upload-zone input { display: none; }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        .upload-text { color: #888; }
        .upload-text span { color: #0ea5e9; }

        .jobs { margin-top: 2rem; }
        .job {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .job-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .job-name { font-weight: 500; }
        .job-status { font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 9999px; }
        .job-status.pending { background: #1e3a5f; color: #60a5fa; }
        .job-status.processing { background: #422006; color: #fbbf24; }
        .job-status.completed { background: #052e16; color: #4ade80; }
        .job-status.failed { background: #450a0a; color: #f87171; }

        /* Progress bar */
        .progress-container {
            background: #1a1a1a;
            border-radius: 4px;
            height: 6px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #0ea5e9, #22d3ee);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-bar.indeterminate {
            width: 30% !important;
            animation: indeterminate 1.5s infinite ease-in-out;
        }
        @keyframes indeterminate {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(400%); }
        }
        .progress-text {
            font-size: 0.75rem;
            color: #888;
            display: flex;
            justify-content: space-between;
        }
        .progress-text .step { color: #aaa; }

        .papers { margin-top: 2rem; }
        .papers h2 { font-size: 1rem; color: #888; margin-bottom: 1rem; }
        .paper-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .paper {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .paper:hover {
            border-color: #444;
            background: #222;
        }

        .chat-link {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #0ea5e9, #06b6d4);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .chat-link:hover {
            background: linear-gradient(135deg, #0284c7, #0891b2);
            transform: translateY(-1px);
        }
        .chat-link-container {
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="particles" id="particles"></div>

    <div class="container">
        <h1>PDF Preprocessor</h1>
        <p class="subtitle">Upload PDFs to extract text and figures for the RAG agent</p>

        <div class="upload-zone" id="dropzone">
            <input type="file" id="fileInput" accept=".pdf" multiple>
            <div class="upload-icon">📄</div>
            <p class="upload-text">Drop PDF files here or <span>click to browse</span></p>
        </div>

        <div class="jobs" id="jobs"></div>

        <div class="papers">
            <h2>Processed Papers</h2>
            <div class="paper-list" id="paperList"></div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const jobsDiv = document.getElementById('jobs');
        const paperList = document.getElementById('paperList');
        const particles = document.getElementById('particles');

        const icons = ['📄', '🖼️', '📊', '📈', '∑', 'ƒ', '∫', '📋', '🔢', '📝'];
        let processingCount = 0;

        function initParticles() {
            for (let i = 0; i < 15; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                p.textContent = icons[Math.floor(Math.random() * icons.length)];
                p.style.left = Math.random() * 100 + '%';
                p.style.animationDelay = Math.random() * 20 + 's';
                p.style.animationDuration = (15 + Math.random() * 10) + 's';
                particles.appendChild(p);
            }
        }
        initParticles();

        function updateParticles() {
            particles.classList.toggle('active', processingCount > 0);
        }

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

        async function handleFiles(files) {
            for (const file of files) {
                if (!file.name.endsWith('.pdf')) continue;
                const formData = new FormData();
                formData.append('file', file);
                try {
                    const resp = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await resp.json();
                    if (data.job_id) pollJob(data.job_id, file.name);
                } catch (err) {
                    console.error('Upload failed:', err);
                }
            }
        }

        function pollJob(jobId, filename) {
            processingCount++;
            updateParticles();

            const jobEl = document.createElement('div');
            jobEl.className = 'job';
            jobEl.id = `job-${jobId}`;
            jobEl.innerHTML = `
                <div class="job-header">
                    <span class="job-name">${filename}</span>
                    <span class="job-status pending" id="status-${jobId}">Pending</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" id="progress-${jobId}" style="width: 0%"></div>
                </div>
                <div class="progress-text">
                    <span class="step" id="step-${jobId}">Waiting...</span>
                    <span id="percent-${jobId}">0%</span>
                </div>
                <div class="chat-link-container" id="chat-${jobId}" style="display: none;"></div>
            `;
            jobsDiv.prepend(jobEl);

            const interval = setInterval(async () => {
                const resp = await fetch(`/status/${jobId}`);
                const data = await resp.json();

                const statusEl = document.getElementById(`status-${jobId}`);
                const progressEl = document.getElementById(`progress-${jobId}`);
                const stepEl = document.getElementById(`step-${jobId}`);
                const percentEl = document.getElementById(`percent-${jobId}`);

                statusEl.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
                statusEl.className = `job-status ${data.status.toLowerCase()}`;

                const progress = data.progress || 0;
                const step = data.step || '';

                if (data.status === 'processing') {
                    if (progress > 0) {
                        progressEl.classList.remove('indeterminate');
                        progressEl.style.width = progress + '%';
                    } else {
                        progressEl.classList.add('indeterminate');
                    }
                    stepEl.textContent = step || 'Processing...';
                    percentEl.textContent = progress > 0 ? Math.round(progress) + '%' : '';
                } else if (data.status === 'completed') {
                    progressEl.classList.remove('indeterminate');
                    progressEl.style.width = '100%';
                    stepEl.textContent = 'Complete!';
                    percentEl.textContent = '100%';

                    // Show chat link
                    const chatContainer = document.getElementById(`chat-${jobId}`);
                    chatContainer.style.display = 'block';
                    chatContainer.innerHTML = `<a href="https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A2030&mode=chat&render=interact&assistantId=3aa5e058-25ba-5c4e-9cd4-b994c9cf54f1" target="_blank" class="chat-link">💬 Chat with Agent</a>`;
                } else if (data.status === 'failed') {
                    progressEl.style.width = '0%';
                    progressEl.style.background = '#ef4444';
                    stepEl.textContent = data.error || 'Failed';
                    percentEl.textContent = '';
                }

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(interval);
                    processingCount--;
                    updateParticles();
                    loadPapers();
                }
            }, 1000);
        }

        async function loadPapers() {
            const resp = await fetch('/papers');
            const papers = await resp.json();
            paperList.innerHTML = papers.length
                ? papers.map(p => `<div class="paper">📁 ${p}</div>`).join('')
                : '<div style="color:#666">No papers yet</div>';
        }

        loadPapers();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and start preprocessing."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    pdf_path = RAW_PDFS_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "progress": 0,
        "step": "Queued",
    }

    asyncio.create_task(process_pdf_with_progress(job_id, pdf_path))
    return {"job_id": job_id, "filename": file.filename}


async def process_pdf_with_progress(job_id: str, pdf_path: Path):
    """Run preprocessing with progress tracking."""
    job = jobs[job_id]
    job["status"] = "processing"

    try:
        pdf_name = pdf_path.stem

        # Setup directories
        images_dir = IMAGES_DIR / pdf_name
        paper_dir = PAPERS_DIR / pdf_name
        figures_dir = paper_dir / "figures"

        images_dir.mkdir(parents=True, exist_ok=True)
        paper_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Convert PDF to images (10%)
        job["step"] = "Converting PDF to images..."
        job["progress"] = 5

        image_paths = await asyncio.to_thread(
            convert_pdf_to_images, str(pdf_path), str(images_dir), DPI
        )
        total_pages = len(image_paths)
        job["progress"] = 10
        job["step"] = f"Converted {total_pages} pages"

        # Phase 2: Extract text + figures (10% - 95%)
        api_key = get_api_key()
        all_results = []
        total_figures = 0

        async with aiohttp.ClientSession() as session:
            for i, img_path in enumerate(image_paths):
                page_num = i + 1
                job["step"] = f"Extracting page {page_num}/{total_pages}"
                job["progress"] = 10 + (i / total_pages) * 85

                markdown, detected_images = await extract_page(session, img_path, api_key)

                _, num_figs = save_page_markdown(
                    page_num, markdown, detected_images, img_path,
                    paper_dir, figures_dir
                )
                all_results.append((page_num, markdown, detected_images, img_path))
                total_figures += num_figs

        # Phase 3: Save combined markdown (95% - 100%)
        job["step"] = "Saving combined document..."
        job["progress"] = 95

        save_combined_markdown(all_results, paper_dir, figures_dir, pdf_name)

        job["progress"] = 100
        job["step"] = f"Done! {total_pages} pages, {total_figures} figures"
        job["status"] = "completed"

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)[:100]


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status with progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/papers")
async def list_papers():
    """List all processed papers."""
    papers = []
    if PAPERS_DIR.exists():
        for p in sorted(PAPERS_DIR.iterdir()):
            if p.is_dir():
                papers.append(p.name)
    return papers


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2031)
