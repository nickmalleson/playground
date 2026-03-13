"""
Helper functions for the Jiaqi papers project.
Includes ORCID metadata extraction, LaTeX formatting, translation, and TeX generation.
"""

import re
import ast
import html
import json
import os
import time

import requests
import pandas as pd

# ===========================================================================
# ORCID helpers
# ===========================================================================

DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)


def normalize_doi(doi):
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.strip(" ,.;")


def fetch_full_work(base, put_code):
    url = f"{base}/work/{put_code}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_ext_ids(full):
    ext_ids = full.get("external-ids", {}).get("external-id")
    if ext_ids is None:
        return []
    if isinstance(ext_ids, dict):
        return [ext_ids]
    if isinstance(ext_ids, list):
        return [e for e in ext_ids if isinstance(e, dict)]
    return []


def extract_id(full, id_type):
    id_type = (id_type or "").lower()
    for eid in extract_ext_ids(full):
        typ = (eid.get("external-id-type") or "").lower()
        if typ == id_type:
            return (eid.get("external-id-value") or "").strip()
    return None


def extract_doi_from_ext_ids(full):
    for eid in extract_ext_ids(full):
        if (eid.get("external-id-type") or "").lower() == "doi":
            return normalize_doi(eid.get("external-id-value"))
    return None


def extract_doi_from_citation(full):
    cit = full.get("citation")
    if not isinstance(cit, dict):
        return None
    cit_val = cit.get("citation-value")
    if not isinstance(cit_val, str):
        return None
    m = DOI_RE.search(cit_val)
    if m:
        return normalize_doi(m.group(1))
    return None


def extract_doi_from_url(full):
    url_obj = full.get("url")
    u = url_obj.get("value") if isinstance(url_obj, dict) else url_obj
    if isinstance(u, str) and "doi.org/" in u.lower():
        return normalize_doi(u)
    return None


def extract_pub_date_parts(full):
    year = month = day = None
    pd_node = full.get("publication-date")
    if isinstance(pd_node, dict):
        y = pd_node.get("year")
        m = pd_node.get("month")
        d = pd_node.get("day")
        if isinstance(y, dict):
            year = y.get("value")
        if isinstance(m, dict):
            month = m.get("value")
        if isinstance(d, dict):
            day = d.get("value")
    return year, month, day


def get_doi(full):
    doi = extract_doi_from_ext_ids(full)
    if doi:
        return doi
    doi = extract_doi_from_citation(full)
    if doi:
        return doi
    doi = extract_doi_from_url(full)
    if doi:
        return doi
    return None


def extract_citation_fields(full):
    cit = full.get("citation")
    if isinstance(cit, dict):
        ctype = cit.get("citation-type")
        cval = cit.get("citation-value")
        if isinstance(cval, str):
            return ctype, cval
    return None, None


def extract_authors(full):
    out = []
    contribs = (full.get("contributors") or {}).get("contributor") or []
    if not isinstance(contribs, list):
        contribs = []

    def seq_val(c):
        seq = ((c.get("contributor-attributes") or {}).get("contributor-sequence") or "")
        try:
            return int(seq)
        except Exception:
            return 1_000_000

    contribs_sorted = sorted(contribs, key=seq_val)
    for c in contribs_sorted:
        credit = (c.get("credit-name") or {}).get("value")
        if credit:
            out.append(credit.strip())
            continue
        name = c.get("contributor-name") or {}
        given = (name.get("given-names") or {}).get("value")
        family = (name.get("family-name") or {}).get("value")
        if given or family:
            out.append(" ".join([p for p in [given, family] if p]).strip())
    seen = set()
    ordered = []
    for a in out:
        if a and a not in seen:
            ordered.append(a)
            seen.add(a)
    return ordered


def safe_url(full):
    url_obj = full.get("url")
    u = url_obj.get("value") if isinstance(url_obj, dict) else url_obj
    return u if isinstance(u, str) else None


def orcid_abstract(full):
    sd = full.get("short-description")
    return sd if isinstance(sd, str) and sd.strip() else None


def strip_tags(text):
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def crossref_fetch(doi, contact_email, use_crossref=True):
    if not use_crossref or not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"ORCID-Book-Builder (mailto:{contact_email})"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return {}
        return r.json().get("message", {}) or {}
    except Exception:
        return {}


def merge_with_crossref_if_needed(orcid_meta, cr):
    updated = dict(orcid_meta)
    if not updated.get("abstract"):
        updated["abstract"] = strip_tags(cr.get("abstract"))
    if not updated.get("journal"):
        ct = cr.get("container-title") or []
        if isinstance(ct, list) and ct:
            updated["journal"] = ct[0]
    updated["volume"] = updated.get("volume") or cr.get("volume")
    updated["issue"] = updated.get("issue") or cr.get("issue")
    updated["pages"] = updated.get("pages") or cr.get("page")
    if not updated.get("authors"):
        auths = []
        for a in cr.get("author", []) or []:
            given = a.get("given")
            family = a.get("family")
            name = " ".join([p for p in [given, family] if p]).strip()
            if name:
                auths.append(name)
        if auths:
            updated["authors"] = auths
    if not updated.get("year"):
        dp = ((cr.get("issued") or {}).get("date-parts") or [[]])
        if dp and dp[0]:
            updated["year"] = str(dp[0][0]) if len(dp[0]) >= 1 else None
            updated["month"] = str(dp[0][1]) if len(dp[0]) >= 2 else None
            updated["day"] = str(dp[0][2]) if len(dp[0]) >= 3 else None
    return updated


def fetch_orcid_papers(orcid_id, contact_email, use_crossref=True, sleep=0.2):
    """Fetch all papers from ORCID and return a DataFrame."""
    base = f"https://pub.orcid.org/v3.0/{orcid_id}"
    headers = {"Accept": "application/json"}

    print("Fetching list of works...")
    summary = requests.get(f"{base}/works", headers=headers, timeout=30).json()

    put_codes = []
    for group in summary.get("group", []):
        for w in group.get("work-summary", []):
            put_codes.append(w["put-code"])

    print(f"Found {len(put_codes)} works. Fetching full records...\n")
    records = []

    for pc in put_codes:
        full = fetch_full_work(base, pc)
        title = (full.get("title") or {}).get("title", {}) or {}
        title = title.get("value")
        year, month, day = extract_pub_date_parts(full)
        journal = (full.get("journal-title") or {}).get("value")
        doi = get_doi(full)
        url = safe_url(full)
        issn = extract_id(full, "issn")
        eissn = extract_id(full, "eissn") or extract_id(full, "e-issn")
        isbn = extract_id(full, "isbn")
        citation_type, citation_value = extract_citation_fields(full)
        authors = extract_authors(full)
        abstract = orcid_abstract(full)
        volume = issue = pages = None

        meta = {
            "put_code": pc, "type": full.get("type"),
            "title": title, "subtitle": None,
            "year": year, "month": month, "day": day,
            "journal": journal, "volume": volume, "issue": issue, "pages": pages,
            "doi": doi, "url": url,
            "issn": issn, "eissn": eissn, "isbn": isbn,
            "authors": "; ".join(authors) if authors else None,
            "citation_type": citation_type, "citation_value": citation_value,
            "abstract": abstract,
            "abstract_source": "orcid" if abstract else None,
        }

        if use_crossref and doi:
            cr = crossref_fetch(doi, contact_email, use_crossref)
            if cr:
                before = dict(meta)
                meta = merge_with_crossref_if_needed(meta, cr)
                if meta.get("abstract") and not before.get("abstract"):
                    meta["abstract_source"] = "crossref"

        records.append(meta)
        short_title = title[:60] if title else "No title"
        print(f"Fetched {pc}: {short_title}  --> DOI: {doi}")
        time.sleep(sleep)

    df = pd.DataFrame(records)
    return df


# ===========================================================================
# LaTeX helpers
# ===========================================================================

def latex_escape(text):
    if not isinstance(text, str):
        return ""
    reps = [
        (r'\\', r'\\textbackslash{}'),
        (r'&', r'\&'),
        (r'%', r'\%'),
        (r'\$', r'\$'),
        (r'#', r'\#'),
        (r'_', r'\_'),
        (r'\{', r'\{'),
        (r'\}', r'\}'),
        (r'~', r'\textasciitilde{}'),
        (r'\^', r'\textasciicircum{}'),
    ]
    out = text
    for a, b in reps:
        out = re.sub(a, b, out)
    return out


def normalize_space(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def format_authors(authors_cell, max_authors=20):
    if not isinstance(authors_cell, str) or not authors_cell.strip():
        return ""
    s = authors_cell.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parts = ast.literal_eval(s)
            parts = [normalize_space(a) for a in parts if normalize_space(a)]
        except Exception:
            parts = [normalize_space(a) for a in s.split(";") if normalize_space(a)]
    else:
        parts = [normalize_space(a) for a in s.split(";") if normalize_space(a)]
    if not parts:
        return ""
    if len(parts) > max_authors:
        parts = parts[:max_authors]
        return "; ".join(parts) + "; et al."
    return "; ".join(parts)


def apa_like_citation(row):
    authors = format_authors(row.get("authors", ""))
    title = normalize_space(row.get("title") or "")
    journal = normalize_space(row.get("journal") or "")
    if journal == "nan":
        journal = ""
    volume = row.get("volume")
    issue = row.get("issue")
    pages = normalize_space(str(row.get("pages") or "").strip())
    if pages == "nan":
        pages = ""
    doi = normalize_space(row.get("doi") or "")

    parts = []
    if authors:
        parts.append(latex_escape(authors) + ("" if authors.endswith(".") else "."))
    if row.get("year"):
        v = str(row.get("year") or "").strip()
        if v:
            parts.append(f"({latex_escape(str(int(float(v))))})")
    if title:
        t = latex_escape(title)
        parts.append(t + ("" if t.endswith(".") else "."))

    vip_bits = []
    if journal:
        vip_bits.append(r"\textit{" + latex_escape(journal) + "}")
    if pd.notna(volume) and pd.notna(issue):
        vip_bits.append(latex_escape(f"{volume}({issue})"))
    elif pd.notna(volume):
        vip_bits.append(latex_escape(str(volume)))
    if pages:
        vip_bits.append(latex_escape(pages))
    if vip_bits:
        parts.append(", ".join(vip_bits) + ".")
    if doi:
        parts.append(f"\\url{{https://doi.org/{doi}}}")

    return " ".join(parts).strip()


def abstract_text(row):
    abs_txt = row.get("abstract")
    if isinstance(abs_txt, str) and abs_txt.strip():
        decoded = html.unescape(abs_txt)
        return latex_escape(normalize_space(decoded))
    return ""


# ===========================================================================
# Translation (Google Translate via deep-translator, with JSON file cache)
# ===========================================================================

TRANSLATION_CACHE_PATH = "translations_cache.json"


def _load_cache():
    if os.path.exists(TRANSLATION_CACHE_PATH):
        with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_to_chinese(text):
    """Translate English text to Simplified Chinese, with caching."""
    if not text or not text.strip():
        return ""

    text = text.strip()
    cache = _load_cache()
    if text in cache:
        return cache[text]

    from deep_translator import GoogleTranslator

    # Google Translate has a ~5000 char limit per request; chunk if needed
    max_chunk = 4500
    if len(text) <= max_chunk:
        result = GoogleTranslator(source="en", target="zh-CN").translate(text)
    else:
        # Split on sentence boundaries
        chunks = []
        current = ""
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if len(current) + len(sentence) + 1 > max_chunk:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = (current + " " + sentence).strip()
        if current:
            chunks.append(current)

        translated_chunks = []
        for chunk in chunks:
            translated_chunks.append(
                GoogleTranslator(source="en", target="zh-CN").translate(chunk)
            )
            time.sleep(0.3)
        result = "".join(translated_chunks)

    cache[text] = result
    _save_cache(cache)
    return result


# ===========================================================================
# TeX generation
# ===========================================================================

def write_chapters_tex(df, output_path="chapters.tex"):
    """Write chapters.tex with English and Chinese translations."""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            doi = row.get("doi")
            if pd.isna(doi) or not isinstance(doi, str) or not doi.strip():
                continue

            title = row.get("title") or ""

            # --- Chapter heading (English + Chinese) ---
            title_zh = translate_to_chinese(title)
            f.write(f"\\chapter{{{latex_escape(title)}}}\n")
            if title_zh:
                f.write(f"\\begin{{center}}\\small {title_zh}\\end{{center}}\n")
            f.write("\n")

            # --- Citation (English) ---
            citation_line = apa_like_citation(row)
            if citation_line:
                f.write("\\noindent\\textbf{Citation:} " + citation_line + "\n\n")

            # --- Citation (Chinese) ---
            citation_plain = _plain_citation(row)
            if citation_plain:
                citation_zh = translate_to_chinese(citation_plain)
                if citation_zh:
                    f.write("\\vspace{0.5em}\n")
                    f.write("\\noindent\\textbf{引用:} " + latex_escape(citation_zh) + "\n\n")

            # --- Abstract (English) ---
            abs_txt = abstract_text(row)
            if abs_txt:
                f.write("\\vspace{1.5em}\n")
                f.write("\\noindent\\textbf{Abstract}\\par\n")
                f.write("\\begin{quote}\\small\n")
                f.write(abs_txt + "\n")
                f.write("\\end{quote}\n\n")

                # --- Abstract (Chinese) ---
                raw_abstract = normalize_space(html.unescape(row.get("abstract", "")))
                abs_zh = translate_to_chinese(raw_abstract)
                if abs_zh:
                    f.write("\\noindent\\textbf{摘要}\\par\n")
                    f.write("\\begin{quote}\\small\n")
                    f.write(abs_zh + "\n")
                    f.write("\\end{quote}\n\n")

            # --- PDF Include ---
            pdf_file = f"pdfs/{doi.replace('/', '_')}.pdf"
            if os.path.exists(pdf_file):
                f.write("\\vspace{0.5em}\n")
                f.write(f"\\includepdf[pages=-]{{{pdf_file}}}\n\n")
            else:
                f.write("\\vspace{0.5em}\n")

    print(f"Wrote {output_path}")


def _plain_citation(row):
    """Build a plain-text citation string for translation (no LaTeX markup)."""
    authors = format_authors(row.get("authors", ""))
    title = normalize_space(row.get("title") or "")
    journal = normalize_space(row.get("journal") or "")
    if journal == "nan":
        journal = ""
    volume = row.get("volume")
    issue = row.get("issue")
    pages = normalize_space(str(row.get("pages") or "").strip())
    if pages == "nan":
        pages = ""
    doi = normalize_space(row.get("doi") or "")

    parts = []
    if authors:
        parts.append(authors + ("" if authors.endswith(".") else "."))
    if row.get("year"):
        v = str(row.get("year") or "").strip()
        if v:
            parts.append(f"({str(int(float(v)))})")
    if title:
        parts.append(title + ("" if title.endswith(".") else "."))
    vip_bits = []
    if journal:
        vip_bits.append(journal)
    if pd.notna(volume) and pd.notna(issue):
        vip_bits.append(f"{volume}({issue})")
    elif pd.notna(volume):
        vip_bits.append(str(volume))
    if pages:
        vip_bits.append(pages)
    if vip_bits:
        parts.append(", ".join(vip_bits) + ".")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    return " ".join(parts).strip()
