#!/usr/bin/env python3
"""
scrape_who_fact_sheets.py

Scrapes WHO Fact Sheets (https://www.who.int/news-room/fact-sheets) and produces a JSONL
file with chunked documents ready for RAG ingestion.

Outputs: who_diseases_Full_Catalogue.jsonl

Usage:
    python scrape_who_fact_sheets.py

Notes:
 - Respect robots.txt and site scraping rules.
 - The script uses tiktoken for token-accurate chunking with sentence-aware splitting.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
from pathlib import Path
from datetime import datetime

# Ensure tiktoken is available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install it for better chunking: pip install tiktoken")

# NLTK for sentence splitting
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize
    except Exception as e:
        print(f"Warning: nltk punkt tokenizer not available: {e}. Sentence-based chunking may fail.")
        sent_tokenize = None

BASE_INDEX = "https://www.who.int/news-room/fact-sheets"
BASE_DOMAIN = "https://www.who.int"
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "who_diseases_Full_Catalogue.jsonl"
MAX_PAGES = 1000            # how many disease pages to scrape
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-Scraper/1.0; +https://example.com/)"
}
REQUEST_DELAY = 1.0       # seconds between requests (politeness)
MAX_CHUNK_TOKENS = 450    # target chunk size for tiktoken
MIN_CHUNK_TOKENS = 200    # minimum acceptable chunk size
MIN_CHUNK_TOKENS_STRICT = 50  # absolute minimum - chunks below this will always be merged
CITATION_HEAVY_THRESHOLD = 0.3  # if >30% of chunk is citations, mark as citation-heavy

def fetch_url(url):
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"[ERR] Failed to fetch {url}: {e}")
        return None

def find_fact_sheet_links(index_html):
    soup = BeautifulSoup(index_html, "lxml")
    links = set()
    # WHO fact sheets often use links like /news-room/fact-sheets/detail/<slug>
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/news-room/fact-sheets/detail/" in href:
            full = urljoin(BASE_DOMAIN, href)
            links.add(full)
    # Return as list
    return sorted(list(links))

def clean_text(text):
    """
    Comprehensive text cleaning function for embedding preparation.
    Removes reference markers, citations, normalizes whitespace, and converts newlines to spaces.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string ready for embedding
    """
    if not text:
        return ""
    
    # Remove PMID citations: PMID: 34106927, PMCID: PMC8189560
    text = re.sub(r"\b(PMID|PMCID):\s*[\w\d-]+\s*;?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPMID:\s*[\w\d-]+;?\s*", "", text, flags=re.IGNORECASE)
    
    # Remove parenthetical citations with multiple numbers: (6,7,8), (4, 5), (8,10), (6, 7, 8)
    text = re.sub(r"\(\s*\d+\s*(?:[,\s]+\s*\d+\s*)+\)", "", text)
    
    # Remove standalone number references at end of lines: " 1\n", " . 1\n"
    # Pattern: space(s) or period+space, followed by 1-2 digits, followed only by whitespace/newline/end
    # This targets reference citations (typically 1-2 digits) while preserving contextual numbers
    # Examples removed: " 1\n", " . 1\n", "text . 1\n"
    # Examples preserved: "20 years", "US$ 553 million", "in 2020"
    text = re.sub(r"(?<![a-zA-Z0-9$%])\s*\.?\s+(\d{1,2})\s*(?=\n|$)", "", text)
    
    # Remove reference markers: [1], [2], (1), (2) - single numbers in brackets/parentheses
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*\d+\s*\)", "", text)
    
    # Remove WHO citations and similar patterns: (WHO, 2024), (WHO 2024)
    text = re.sub(r"\(\s*WHO[^\)]*\)", "", text)
    
    # Remove academic citations: Smith et al., 2024, Author et al. (2024)
    text = re.sub(r"\b[A-Z][a-z]+ et al\.,? \d{4}\b", "", text)
    text = re.sub(r"\b[A-Z][a-z]+ et al\.\s*\(\s*\d{4}\s*\)", "", text)
    
    # Normalize line endings: convert \r\n to \n
    text = re.sub(r"\r\n", "\n", text)
    
    # Replace all newlines (\n) with spaces for better embedding
    # This helps with semantic search as newlines can cause issues in embeddings
    text = re.sub(r"\n+", " ", text)
    
    # Normalize multiple spaces/tabs to single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def find_main_content(soup):
    """
    Finds the main content area in the HTML soup.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        BeautifulSoup element containing main content
    """
    # Remove navigation and non-content elements
    for tag in soup(["header", "footer", "nav", "script", "style", "aside", "form"]):
        tag.decompose()
    
    # Try to find main content area
    main = soup.find("main")
    if main is None:
        # Fallback to article or content container
        main = soup.find("article") or soup.find(class_=re.compile("content|main|primary", re.I)) or soup
    
    return main


def extract_heading_sections(main_element):
    """
    Extracts sections based on headings from the main content element.
    
    Args:
        main_element: BeautifulSoup element containing main content
        
    Returns:
        List of (section_title, section_text) tuples
    """
    sections = []
    current_title = None
    current_paras = []
    
    # Iterate through top-level children of main
    for el in main_element.find_all(recursive=False):
        # Check if this is a heading
        if el.name and re.match(r"h[1-6]", el.name, re.I):
            # Save previous section if exists
            if current_title and current_paras:
                text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in current_paras)
                sections.append((current_title, clean_text(text)))
            current_title = el.get_text(separator=" ", strip=True)
            current_paras = []
        else:
            # Check if element contains nested headings
            if el.find(re.compile(r"h[2-4]")):
                # Handle inner headings recursively
                for child in el.descendants:
                    if getattr(child, "name", None) and re.match(r"h[1-6]", child.name, re.I):
                        # Save previous section
                        if current_title and current_paras:
                            text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in current_paras)
                            sections.append((current_title, clean_text(text)))
                        current_title = child.get_text(separator=" ", strip=True)
                        current_paras = []
                    elif getattr(child, "name", None) in ("p", "div"):
                        if child.get_text(strip=True):
                            current_paras.append(child)
            else:
                # Collect paragraph-like elements
                if el.name in ("p", "div", "section"):
                    if el.get_text(strip=True):
                        # Find previous heading if no current title
                        if current_title is None:
                            prev_heading = el.find_previous(re.compile(r"h[1-4]"))
                            if prev_heading:
                                current_title = prev_heading.get_text(separator=" ", strip=True)
                        current_paras.append(el)
                else:
                    # Fallback: collect inner paragraph tags
                    ps = el.find_all("p")
                    for p in ps:
                        if p.get_text(strip=True):
                            if current_title is None:
                                prev_heading = el.find_previous(re.compile(r"h[1-4]"))
                                if prev_heading:
                                    current_title = prev_heading.get_text(separator=" ", strip=True)
                            current_paras.append(p)
    
    # Save last section
    if current_title and current_paras:
        text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in current_paras)
        sections.append((current_title, clean_text(text)))
    
    return sections


def get_fallback_content(main_element, soup):
    """
    Gets fallback content when no structured sections are found.
    
    Args:
        main_element: BeautifulSoup element containing main content
        soup: BeautifulSoup object for full page fallback
        
    Returns:
        List with single (title, text) tuple
    """
    paragraphs = main_element.find_all("p")
    if paragraphs:
        text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return [("Content", clean_text(text))]
    else:
        # Last resort: whole page text
        all_text = soup.get_text(separator="\n", strip=True)
        return [("Content", clean_text(all_text))]


def merge_short_sections(sections, min_words=30):
    """
    Merges sections that are too short with the previous section.
    
    Args:
        sections: List of (title, text) tuples
        min_words: Minimum word count to keep as separate section
        
    Returns:
        List of merged (title, text) tuples
    """
    if not sections:
        return sections
    
    merged = []
    for title, text in sections:
        word_count = len(text.split())
        if word_count < min_words and merged:
            # Append to previous section
            prev_title, prev_text = merged[-1]
            merged[-1] = (prev_title, prev_text + "\n\n" + text)
        else:
            merged.append((title, text))
    
    return merged


def extract_sections_from_html(html, page_url):
    """
    Extract sections based on headings. Returns list of (section_title, section_text).
    If headings are not helpful, returns a single 'content' section.
    
    This function orchestrates the extraction process by calling helper functions.
    
    Args:
        html: HTML string to parse
        page_url: URL of the page (for reference, not currently used)
        
    Returns:
        List of (section_title, section_text) tuples
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Find main content area
    main = find_main_content(soup)
    
    # Extract sections based on headings
    sections = extract_heading_sections(main)
    
    # Fallback if no sections found
    if not sections:
        sections = get_fallback_content(main, soup)
    
    # Merge short sections
    sections = merge_short_sections(sections, min_words=30)
    
    return sections

def is_citation_heavy(text):
    """
    Detects if a chunk is mostly citations/references.
    
    Args:
        text: Text chunk to analyze
        
    Returns:
        True if chunk is citation-heavy (mostly references), False otherwise
    """
    if not text or len(text.strip()) < 20:
        return True  # Very short chunks are likely just citations
    
    # Count citation patterns
    citation_patterns = [
        r"PMID:\s*\d+",
        r"PMCID:\s*\w+",
        r"doi:\s*[0-9.]+/[^\s]+",
        r"ISBN[:\s]+\d+",
        r"https?://[^\s]+",  # URLs
        r"\.\s*\d{1,2}\s*$",  # Standalone numbers at end
        r"\(\s*\d+[,\s]+\d+\s*\)",  # Multiple citations
    ]
    
    total_chars = len(text)
    citation_chars = 0
    
    # Count characters in citation patterns
    for pattern in citation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            citation_chars += len(match.group())
    
    # Check for high density of citation-like patterns
    citation_density = citation_chars / total_chars if total_chars > 0 else 0
    
    # Check if text is mostly references (author names, titles, etc.)
    lines = text.split('\n')
    if len(lines) > 2:
        # If multiple short lines, likely a reference list
        avg_line_length = sum(len(line.strip()) for line in lines) / len(lines)
        if avg_line_length < 50:
            citation_density += 0.2
    
    return citation_density > CITATION_HEAVY_THRESHOLD


def merge_small_chunks(chunks, chunks_raw=None, min_tokens_strict=MIN_CHUNK_TOKENS_STRICT, 
                       min_tokens=MIN_CHUNK_TOKENS, max_tokens=MAX_CHUNK_TOKENS, 
                       encoding_name="cl100k_base"):
    """
    Aggressively merge small chunks and handle citation-heavy chunks.
    
    Args:
        chunks: List of cleaned text chunks
        chunks_raw: List of raw text chunks (before cleaning) for citation detection
        min_tokens_strict: Absolute minimum tokens - chunks below this will always be merged
        min_tokens: Minimum acceptable chunk size
        max_tokens: Maximum chunk size
        encoding_name: Tiktoken encoding name
        
    Returns:
        List of merged chunks
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: merge based on word count
        merged = []
        buffer = ""
        for chunk in chunks:
            words = len(chunk.split())
            if words < min_tokens_strict // 2:
                buffer = (buffer + " " + chunk).strip() if buffer else chunk
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(chunk)
        if buffer:
            if merged:
                merged[-1] = (merged[-1] + " " + buffer).strip()
            else:
                merged.append(buffer)
        return merged
    
    enc = tiktoken.get_encoding(encoding_name)
    filtered_chunks = []
    
    # Use raw chunks for citation detection if available, otherwise use cleaned chunks
    chunks_for_detection = chunks_raw if chunks_raw and len(chunks_raw) == len(chunks) else chunks
    
    # First pass: filter out citation-heavy chunks and categorize by size
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(enc.encode(chunk))
        raw_chunk = chunks_for_detection[i] if chunks_for_detection else chunk
        
        # Skip citation-heavy chunks (they dilute embeddings) - check raw chunk
        if is_citation_heavy(raw_chunk):
            continue  # Skip these chunks
        
        # If chunk is too small, add to buffer for merging
        if chunk_tokens < min_tokens_strict:
            # Keep very small chunks only if they're not citation-heavy
            filtered_chunks.append(("SMALL", chunk, chunk_tokens))
        elif chunk_tokens < min_tokens:
            filtered_chunks.append(("MEDIUM", chunk, chunk_tokens))
        else:
            filtered_chunks.append(("NORMAL", chunk, chunk_tokens))
    
    # Second pass: merge small chunks with adjacent ones
    merged_chunks = []
    buffer = ""
    buffer_tokens = 0
    buffer_type = None
    
    for chunk_type, chunk, chunk_tokens in filtered_chunks:
        if chunk_type == "SMALL" or chunk_type == "MEDIUM":
            # Try to merge with buffer
            if buffer:
                combined_tokens = buffer_tokens + chunk_tokens
                if combined_tokens <= max_tokens:
                    buffer = (buffer + " " + chunk).strip()
                    buffer_tokens = combined_tokens
                    continue
                else:
                    # Can't merge, finalize buffer first
                    if buffer_tokens >= min_tokens_strict:
                        merged_chunks.append(clean_text(buffer))
                    buffer = chunk
                    buffer_tokens = chunk_tokens
            else:
                buffer = chunk
                buffer_tokens = chunk_tokens
                buffer_type = chunk_type
        else:
            # Normal-sized chunk
            if buffer:
                # Try to merge buffer with this chunk
                combined_tokens = buffer_tokens + chunk_tokens
                if combined_tokens <= max_tokens and buffer_type in ("SMALL", "MEDIUM"):
                    merged_chunks.append(clean_text(buffer + " " + chunk))
                    buffer = ""
                    buffer_tokens = 0
                else:
                    # Can't merge, add buffer separately if large enough
                    if buffer_tokens >= min_tokens_strict:
                        merged_chunks.append(clean_text(buffer))
                    merged_chunks.append(clean_text(chunk))
                    buffer = ""
                    buffer_tokens = 0
            else:
                merged_chunks.append(clean_text(chunk))
    
    # Handle remaining buffer
    if buffer:
        if buffer_tokens >= min_tokens_strict:
            merged_chunks.append(clean_text(buffer))
        elif merged_chunks:
            # Merge with last chunk if possible
            last_chunk = merged_chunks[-1]
            last_tokens = len(enc.encode(last_chunk))
            if last_tokens + buffer_tokens <= max_tokens:
                merged_chunks[-1] = clean_text(last_chunk + " " + buffer)
            else:
                # Can't merge, but keep it if it's reasonable
                if buffer_tokens >= min_tokens_strict // 2:
                    merged_chunks.append(clean_text(buffer))
    
    return merged_chunks if merged_chunks else []


def chunk_text_token_sentences(text, max_tokens=MAX_CHUNK_TOKENS, min_tokens=MIN_CHUNK_TOKENS, encoding_name="cl100k_base"):
    """
    Sentence-aware token-based chunking using tiktoken.
    This is the improved chunking function that preserves sentence boundaries.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum acceptable chunk size
        encoding_name: Tiktoken encoding to use
        
    Returns:
        List of text chunks
    """
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding(encoding_name)
    else:
        # Fallback to word-based chunking if tiktoken not available
        return chunk_text_word_based(text, min_words=min_tokens // 2, max_words=max_tokens // 2)
    
    # Try NLTK sentence tokenization
    if sent_tokenize:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback split on periods
            sentences = text.split(". ")
            sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
    else:
        # Fallback split
        sentences = text.split(". ")
        sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
    
    chunks_raw = []  # Store raw chunks before cleaning
    chunks = []      # Store cleaned chunks
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = len(enc.encode(sent))
        
        # If adding this sentence would exceed max_tokens, finalize current chunk
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            raw_chunk = " ".join(current_chunk)
            chunks_raw.append(raw_chunk)
            chunks.append(clean_text(raw_chunk))
            current_chunk = [sent]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
    
    # Add remaining chunk
    if current_chunk:
        raw_chunk = " ".join(current_chunk)
        chunks_raw.append(raw_chunk)
        chunks.append(clean_text(raw_chunk))
    
    # Use improved merging function to handle small chunks and citation-heavy chunks
    # Pass raw chunks for citation detection, cleaned chunks for merging
    merged_chunks = merge_small_chunks(
        chunks, 
        chunks_raw=chunks_raw,  # Pass raw chunks for citation detection
        min_tokens_strict=MIN_CHUNK_TOKENS_STRICT,
        min_tokens=min_tokens, 
        max_tokens=max_tokens,
        encoding_name=encoding_name
    )
    
    return merged_chunks if merged_chunks else [clean_text(text)]


def chunk_text_word_based(text, min_words=100, max_words=225):
    """Fallback word-based chunking when tiktoken is not available."""
    if not sent_tokenize:
        # Last resort: split on periods
        sentences = text.split(". ")
        sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
    else:
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = text.split(". ")
            sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
    
    chunks = []
    current = []
    current_words = 0
    
    for s in sentences:
        w = len(s.split())
        if current_words + w >= max_words and current:
            chunks.append(clean_text(" ".join(current)))
            current = [s]
            current_words = w
        else:
            current.append(s)
            current_words += w
    
    if current:
        chunks.append(clean_text(" ".join(current)))
    
    # Merge tiny chunks
    merged = []
    buffer = ""
    for chunk in chunks:
        words = len(chunk.split())
        if words < min_words:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
        else:
            if buffer:
                merged.append(clean_text(buffer))
                buffer = ""
            merged.append(clean_text(chunk))
    
    if buffer:
        merged.append(clean_text(buffer))
    
    return merged if merged else [clean_text(text)]

def slug_from_url(url):
    parsed = urlparse(url)
    # path like /news-room/fact-sheets/detail/dengue-and-severe-dengue
    parts = [p for p in parsed.path.split("/") if p]
    if parts:
        return parts[-1]
    return re.sub(r"\W+", "-", url)

def build_id(title, section, chunk_idx):
    safe_title = re.sub(r"\W+", "_", title.lower()).strip("_")
    safe_section = re.sub(r"\W+", "_", section.lower()).strip("_")
    return f"{safe_title}_{safe_section}_chunk_{chunk_idx}"

def generate_summary_report(stats, output_file, start_time):
    """
    Generate a summary report of the scraping process.
    
    Args:
        stats: Dictionary containing statistics
        output_file: Path to output file
        start_time: Start time of the process
    """
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    report = f"""
{'='*60}
SCRAPING SUMMARY REPORT
{'='*60}
Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Processing Time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)

STATISTICS:
-----------
Total Pages Processed: {stats['pages_processed']}
Total Pages Failed: {stats['pages_failed']}
Total Chunks Created: {stats['total_chunks']}
Average Chunks per Page: {stats['total_chunks'] / max(stats['pages_processed'], 1):.2f}

CHUNK STATISTICS:
-----------------
Average Chunk Size: {stats['avg_chunk_size']:.2f} tokens
Smallest Chunk: {stats['min_chunk_size']} tokens
Largest Chunk: {stats['max_chunk_size']} tokens
Total Chunks: {stats['total_chunks']}

CHUNK QUALITY IMPROVEMENTS:
---------------------------
Citation-heavy chunks filtered: Citation-heavy chunks (with >30% references) 
  are automatically filtered to improve embedding quality.
Small chunks merged: Chunks below {MIN_CHUNK_TOKENS_STRICT} tokens are 
  merged with adjacent chunks to ensure minimum quality (target: {MIN_CHUNK_TOKENS} tokens).

FAILED URLS ({len(stats['failed_urls'])}):
{'----------------------------------------'}
"""
    
    if stats['failed_urls']:
        for url, error in list(stats['failed_urls'].items())[:10]:  # Show first 10
            report += f"  - {url}\n    Error: {error}\n"
        if len(stats['failed_urls']) > 10:
            report += f"  ... and {len(stats['failed_urls']) - 10} more\n"
    else:
        report += "  None\n"
    
    report += f"\nOUTPUT FILE:\n------------\n{output_file}\n"
    report += f"{'='*60}\n"
    
    print(report)
    
    # Save report to file
    report_file = DATA_DIR / "scraping_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Summary report saved to: {report_file}")


def main():
    """Main scraping function."""
    start_time = datetime.now()
    
    # Initialize statistics
    stats = {
        'pages_processed': 0,
        'pages_failed': 0,
        'total_chunks': 0,
        'chunks_filtered_citation_heavy': 0,
        'chunks_merged_small': 0,
        'failed_urls': {},
        'chunk_sizes': [],
        'avg_chunk_size': 0,
        'min_chunk_size': float('inf'),
        'max_chunk_size': 0
    }
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Fetching WHO fact sheets index...")
    index_html = fetch_url(BASE_INDEX)
    if not index_html:
        print("Failed to fetch WHO index page. Exiting.")
        return
    
    links = find_fact_sheet_links(index_html)
    print(f"Found {len(links)} fact-sheet links on index (unique).")
    if not links:
        print("No links found. Exiting.")
        return
    
    # Limit to MAX_PAGES
    links = links[:MAX_PAGES]
    print(f"Processing up to {len(links)} pages...\n")
    
    # Use context manager for file operations
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for link in tqdm(links, desc="Scraping pages"):
            html = fetch_url(link)
            if not html:
                stats['pages_failed'] += 1
                stats['failed_urls'][link] = "Failed to fetch HTML"
                time.sleep(REQUEST_DELAY)
                continue
            
            try:
                # Extract title from page or from URL
                soup = BeautifulSoup(html, "lxml")
                title_tag = soup.find(["h1", "h2"])
                title = title_tag.get_text(strip=True) if title_tag else slug_from_url(link).replace("-", " ").title()
                
                sections = extract_sections_from_html(html, link)
                
                # Extract category from breadcrumbs
                category = None
                bc = soup.find(class_=re.compile("breadcrumb|breadcrumbs|page-breadcrumb", re.I))
                if bc:
                    category = bc.get_text(separator=" > ", strip=True)
                
                # Process each section
                for sec_title, sec_text in sections:
                    if not sec_text or len(sec_text.strip()) < 50:
                        continue
                    
                    # Chunk text using sentence-aware chunking
                    chunks = chunk_text_token_sentences(sec_text, max_tokens=MAX_CHUNK_TOKENS, min_tokens=MIN_CHUNK_TOKENS)
                    
                    # Write chunks to file
                    for i, chunk in enumerate(chunks, start=1):
                        # Calculate chunk size for statistics
                        if TIKTOKEN_AVAILABLE:
                            chunk_tokens = len(tiktoken.get_encoding("cl100k_base").encode(chunk))
                        else:
                            chunk_tokens = len(chunk.split()) * 1.3  # Approximate
                        
                        stats['chunk_sizes'].append(chunk_tokens)
                        stats['min_chunk_size'] = min(stats['min_chunk_size'], chunk_tokens)
                        stats['max_chunk_size'] = max(stats['max_chunk_size'], chunk_tokens)
                        
                        record = {
                            "id": build_id(title, sec_title, i),
                            "title": title,
                            "section": sec_title,
                            "chunk_id": i,
                            "text": chunk,
                            "source_url": link,
                            "source": "WHO",
                            "category": category or ""
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats['total_chunks'] += 1
                
                stats['pages_processed'] += 1
                
            except Exception as e:
                stats['pages_failed'] += 1
                stats['failed_urls'][link] = str(e)
                print(f"\nError processing {link}: {e}")
            
            # Politeness delay
            time.sleep(REQUEST_DELAY)
    
    # Calculate average chunk size
    if stats['chunk_sizes']:
        stats['avg_chunk_size'] = sum(stats['chunk_sizes']) / len(stats['chunk_sizes'])
    else:
        stats['min_chunk_size'] = 0
    
    # Generate summary report
    generate_summary_report(stats, OUTPUT_FILE, start_time)
    
    print(f"\n✓ Scraping complete! Wrote {stats['total_chunks']} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

