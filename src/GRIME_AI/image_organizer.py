# image_organizer.py
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# 3rd-party
try:
    from PIL import Image, ExifTags
except Exception:
    Image = None
    ExifTags = None

try:
    import piexif
except Exception:
    piexif = None  # we'll guard writes that need piexif

from GRIME_AI.App_logger import info as _info, debug as _debug, err as _err, warn as _warn

# =========================
# Constants / EXIF tags
# =========================
IMG_EXTS: Set[str] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

EXIF_TAGS = {v: k for k, v in getattr(ExifTags, 'TAGS', {}).items()} if ExifTags else {}

# Define Microsoft XP EXIF tags (UTF-16LE encoded)
XPTitle = 0x9C9B
XPComment = 0x9C9C
XPAuthor = 0x9C9D
XPKeywords = 0x9C9E
XPSubject = 0x9C9F

# Filename tag extraction – exact matching only
# Populate CANONICAL_TAGS with your project’s controlled vocabulary (lowercase).
CANONICAL_TAGS: Set[str] = {
    # examples; extend as needed
    "raccoon",
    "beaver",
    "coyote",
    "white-tailed deer",
    "great blue heron",
    "wtd",
    "gbh",  # if you also want to accept the abbrev literally
}

# Abbreviation map (lowercase token -> canonical lowercase tag)
ANIMAL_ABBREV: Dict[str, str] = {
    "gbh": "great blue heron",
    "wtd": "white-tailed deer",
}

STOP_TOKENS: Set[str] = {
    "img", "image", "photo", "dsc", "dcim", "reconyx", "trailcam",
    "cam", "camera", "copy", "edited", "final"
}

DATE_PATTERNS = [
    re.compile(r"^\d{8}$"),             # YYYYMMDD
    re.compile(r"^\d{4}-\d{2}-\d{2}$"), # YYYY-MM-DD
]
TIME_PATTERNS = [
    re.compile(r"^\d{4}$"),             # HHMM
    re.compile(r"^\d{6}$"),             # HHMMSS
    re.compile(r"^\d{2}-\d{2}-\d{2}$"), # HH-MM-SS
]


# =========================
# Utility encoders/decoders
# =========================
def _encode_xp(s: str) -> bytes:
    return (s or "").encode("utf-16le") + b"\x00\x00"

def _decode_xp(value) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode('utf-16le', errors='ignore').rstrip('\x00')
        if isinstance(value, (list, tuple)):
            return bytes(value).decode('utf-16le', errors='ignore').rstrip('\x00')
        if isinstance(value, str):
            return value
    except Exception:
        return ""
    return ""


# =========================
# XMP read (lightweight)
# =========================
def read_xmp_chunks(path: Path) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            data = f.read()
        start = data.find(b"<x:xmpmeta")
        if start == -1: return None
        end = data.find(b"</x:xmpmeta>", start)
        if end == -1: return None
        return data[start:end+12].decode('utf-8', errors='ignore')
    except Exception as e:
        _debug(f"read_xmp_chunks error for {path}: {e}")
        return None

def parse_xmp_for_fields(xmp_xml: str) -> Dict[str, object]:
    out = {"Title": "", "Subject": "", "Comments": "", "Tags": [], "Rights": ""}
    try:
        import xml.etree.ElementTree as ET
        ns = {
            'dc': 'http://purl.org/dc/elements/1.1/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/'
        }
        root = ET.fromstring(xmp_xml)
        g = lambda p: root.find(p, ns)

        t = g('.//dc:title//rdf:Alt/rdf:li')
        if t is not None and t.text: out["Title"] = t.text.strip()
        d = g('.//dc:description//rdf:Alt/rdf:li')
        if d is not None and d.text: out["Comments"] = d.text.strip()
        bag = g('.//dc:subject/rdf:Bag')
        if bag is not None:
            tags = []
            for li in bag.findall('rdf:li', ns):
                if li.text: tags.append(li.text.strip())
            out["Tags"] = tags
        lab = g('.//xmp:Label')
        if lab is not None and lab.text: out["Subject"] = lab.text.strip()
        r = g('.//dc:rights//rdf:Alt/rdf:li')
        if r is not None and r.text: out["Rights"] = r.text.strip()
    except Exception as e:
        _debug(f"parse_xmp_for_fields error: {e}")
    return out


# =========================
# Metadata read
# =========================
def read_image_metadata(path: Path) -> Dict[str, object]:
    """
    Returns dict with (among others):
      - DateTimeFromMetadata (YYYY-MM-DD HH:MM:SS) or ""
      - HasSeconds (bool): whether EXIF originally had seconds precision
      - Title, Subject, Comments
      - Tags (list[str])
      - Exif (raw map)
      - CopyrightExisting
    """
    meta: Dict[str, object] = {
        "DateTimeFromMetadata": "",
        "HasSeconds": False,
        "Title": "",
        "Subject": "",
        "Comments": "",
        "Tags": [],
        "Exif": {},
        "CopyrightExisting": ""
    }
    exif_dt = None
    try:
        if Image is not None:
            with Image.open(str(path)) as img:
                exif_raw = img._getexif() or {}
                for tag_id, val in exif_raw.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    meta["Exif"][name] = val

                # Prefer DateTimeOriginal, then Digitized, then DateTime
                for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    if key in meta["Exif"]:
                        dt_str = meta["Exif"][key]
                        if isinstance(dt_str, bytes): dt_str = _decode_xp(dt_str)
                        if isinstance(dt_str, str):
                            if re.search(r"\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}$", dt_str.strip()):
                                meta["HasSeconds"] = True
                            try:
                                dt_norm = dt_str.replace(':', '-', 2)
                                exif_dt = datetime.strptime(dt_norm, "%Y-%m-%d %H:%M:%S")
                                break
                            except Exception:
                                pass

                meta["Title"]    = _decode_xp(meta["Exif"].get("XPTitle", "")) or meta["Exif"].get("ImageDescription", "")
                meta["Comments"] = _decode_xp(meta["Exif"].get("XPComment", ""))
                meta["Subject"]  = _decode_xp(meta["Exif"].get("XPSubject", ""))

                tags = _decode_xp(meta["Exif"].get("XPKeywords", ""))
                if tags:
                    parts = re.split(r'[;,]\s*', tags)
                    meta["Tags"] = [p for p in parts if p]

                cpy = meta["Exif"].get("Copyright")
                if isinstance(cpy, bytes): cpy = _decode_xp(cpy)
                if isinstance(cpy, str): meta["CopyrightExisting"] = cpy.strip()

                if hasattr(img, "info") and isinstance(img.info, dict):
                    if not meta["Comments"] and isinstance(img.info.get("Comment"), str):
                        meta["Comments"] = img.info["Comment"]
                    if not meta["Subject"] and isinstance(img.info.get("Subject"), str):
                        meta["Subject"] = img.info["Subject"]
                    if not meta["CopyrightExisting"] and isinstance(img.info.get("copyright"), str):
                        meta["CopyrightExisting"] = img.info["copyright"].strip()
    except Exception as e:
        _debug(f"read_image_metadata EXIF error for {path}: {e}")

    try:
        x = read_xmp_chunks(path)
        if x:
            xm = parse_xmp_for_fields(x)
            if not meta["Title"] and xm.get("Title"): meta["Title"] = xm["Title"]
            if not meta["Subject"] and xm.get("Subject"): meta["Subject"] = xm["Subject"]
            if not meta["Comments"] and xm.get("Comments"): meta["Comments"] = xm["Comments"]
            if xm.get("Tags"):
                existing = {t.lower() for t in meta["Tags"]}
                for t in xm["Tags"]:
                    if t.lower() not in existing:
                        meta["Tags"].append(t)
            if not meta["CopyrightExisting"] and xm.get("Rights"):
                meta["CopyrightExisting"] = xm["Rights"]
    except Exception as e:
        _debug(f"read_image_metadata XMP error for {path}: {e}")

    if exif_dt:
        meta["DateTimeFromMetadata"] = exif_dt.strftime("%Y-%m-%d %H:%M:%S")
    return meta


# =========================
# Discovery & file ops
# =========================
def iter_images(src: Path, recursive: bool) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in src.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    else:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    return files

def scan_for_datetime_presence(src_folder: Path, recursive: bool) -> Tuple[List[Path], List[Path]]:
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {src_folder}")
    has_dt, missing_dt = [], []
    for p in iter_images(src_folder, recursive):
        try:
            m = read_image_metadata(p)
            if m.get("DateTimeFromMetadata"):
                has_dt.append(p)
            else:
                missing_dt.append(p)
        except Exception:
            missing_dt.append(p)
    return has_dt, missing_dt

def move_files_to_subfolder(paths: List[Path], base_dir: Path) -> Path:
    """
    Moves paths into a subfolder under base_dir. Always creates the folder.
    """
    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub = base_dir / f"_NoDateTaken_{tag}"
    sub.mkdir(parents=True, exist_ok=True)
    for p in paths:
        try:
            target = sub / p.name
            i = 1
            while target.exists():
                target = sub / f"{p.stem}_{i:03d}{p.suffix}"
                i += 1
            p.rename(target)
            _info(f"Moved (no date): {p.name} -> {target.name}")
        except Exception as e:
            _warn(f"Move failed for {p}: {e}")
    return sub

def estimate_minute_collision_count(files: List[Path], site: str) -> int:
    """
    Estimate how many files would need suffixes if seconds are NOT included.
    We bucket by (YYYYMMDD_HHMM, site) and count collisions (sum of (count-1) per bucket).
    Uses the same timestamp source as organize_images: EXIF DateTime* if present, else file mtime.
    """
    buckets: Dict[Tuple[str, str], int] = {}
    for p in files:
        try:
            meta = read_image_metadata(p)
            dt, _ = choose_dt_for_filename(p, meta)
            key = (dt.strftime("%Y%m%d_%H%M"), (site or "").strip())
            buckets[key] = buckets.get(key, 0) + 1
        except Exception:
            # treat unreadable as unique to avoid over-warning
            pass
    # collisions are (count - 1) per bucket
    return sum(c - 1 for c in buckets.values() if c > 1)

# =========================
# Filename + writes
# =========================
def choose_dt_for_filename(path: Path, meta: Dict[str, object]) -> Tuple[datetime, bool]:
    """
    Returns (dt_for_name, from_metadata_flag)
    """
    if meta.get("DateTimeFromMetadata"):
        try:
            dt = datetime.strptime(meta["DateTimeFromMetadata"], "%Y-%m-%d %H:%M:%S")
            return dt, True
        except Exception:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime), False

def make_filename(dt: datetime, site: str, include_seconds: bool, ext: str) -> str:
    fmt = "%Y%m%d_%H%M%S" if include_seconds else "%Y%m%d_%H%M"
    ts = dt.strftime(fmt)
    site_part = f"_{site.strip()}" if site and site.strip() else ""
    return f"{ts}{site_part}{ext}"

def get_image_size_safe(path: Path) -> Tuple[int, int]:
    try:
        if Image is None:
            return (0, 0)
        with Image.open(str(path)) as im:
            return (int(im.width), int(im.height))
    except Exception:
        return (0, 0)

def append_resolution_suffix(filename: str, w: int, h: int) -> str:
    p = Path(filename)
    return f"{p.stem}_{w}x{h}{p.suffix}"

def append_counter_suffix(filename: str, ctr: int) -> str:
    p = Path(filename)
    return f"{p.stem}_{ctr:02d}{p.suffix}"

def safe_rename(path: Path, new_name: str) -> Path:
    target = path.with_name(new_name)
    if not target.exists():
        path.rename(target); return target
    stem, ext = target.stem, target.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i:03d}{ext}")
        if not cand.exists():
            path.rename(cand); return cand
        i += 1

def write_copyright_exif(path: Path, holder: str, skip_if_exists: bool = True) -> None:
    if not holder or path.suffix.lower() not in {'.jpg', '.jpeg', '.tif', '.tiff'} or piexif is None:
        return
    try:
        exif = piexif.load(str(path))
        if skip_if_exists:
            existing = exif['0th'].get(piexif.ImageIFD.Copyright, b"")
            if isinstance(existing, (bytes, bytearray)) and existing.strip():
                _debug(f"Existing copyright present; skipping write for {path}")
                return
        exif['0th'][piexif.ImageIFD.Copyright] = holder.encode("utf-8", errors="ignore")
        piexif.insert(piexif.dump(exif), str(path))
        _debug(f"Wrote Copyright to {path}")
    except Exception as e:
        _warn(f"Failed to write Copyright EXIF for {path}: {e}")

def compose_final_comments(original_comments: str, title: str, site_info: str) -> str:
    """
    Compose final Comments = [existing Comments][ | Title][\nSite Info].
    """
    parts = []
    if original_comments.strip():
        parts.append(original_comments.strip())
    if title.strip():
        parts.append(title.strip())
    if site_info.strip():
        parts.append(site_info.strip())
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if site_info.strip():
        return " | ".join(parts[:-1]) + "\n" + parts[-1]
    return " | ".join(parts)

def write_comments_and_clear_title(path: Path, meta: Dict[str, object], site_info: str) -> str:
    """
    Build final Comments = [existing Comments][ | Title][\nSite Info],
    clear Title, and write back to EXIF for JPEG/TIFF.
    """
    final_comments = compose_final_comments(
        original_comments=str(meta.get("Comments","")),
        title=str(meta.get("Title","")),
        site_info=site_info or ""
    )
    if path.suffix.lower() in {'.jpg', '.jpeg', '.tif', '.tiff'} and piexif is not None:
        try:
            ex = piexif.load(str(path))
            ex['0th'][XPTitle]   = _encode_xp("")                # clear title
            ex['0th'][XPComment] = _encode_xp(final_comments)    # set comments
            piexif.insert(piexif.dump(ex), str(path))
            _debug(f"Wrote Comments & cleared Title in {path.name}")
        except Exception as e:
            _warn(f"Failed to write comments/title for {path}: {e}")
    return final_comments

def _excel_escape(s: str) -> str:
    """
    Excel formulas escape double quotes by doubling them
    """
    return (s or "").replace('"', '""')

def _excel_hyperlink(full_path: str, display_text: str) -> str:
    """
    Build =HYPERLINK("C:\full\path.jpg","display.jpg")
    """
    return f'=HYPERLINK("{_excel_escape(full_path)}","{_excel_escape(display_text)}")'

# =========================
# Exact tag extraction from filename (position-agnostic)
# =========================
def _split_tokens(name: str) -> List[str]:
    # break CamelCase boundaries
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    # split on anything not a letter
    parts = re.split(r"[^A-Za-z]+", name)
    return [p for p in parts if p]

def _looks_like_date_or_time(tok: str) -> bool:
    for rx in DATE_PATTERNS:
        if rx.match(tok): return True
    for rx in TIME_PATTERNS:
        if rx.match(tok): return True
    return False

def infer_exact_tags_from_filename(path: Path, site: str) -> List[str]:
    """
    Only accept tags that exactly match CANONICAL_TAGS (single, bi-, tri-grams),
    after abbreviation mapping. Site words, numbers and date/time tokens are ignored.
    """
    stem = path.stem
    tokens = _split_tokens(stem)
    low = [t.lower() for t in tokens]

    # remove date/time, digits
    cleaned = [t for t in low if not (t.isdigit() or _looks_like_date_or_time(t))]
    # remove site words anywhere
    site_words = [w.lower() for w in _split_tokens(site or "")]
    cleaned = [t for t in cleaned if t not in site_words]
    # remove stop tokens
    cleaned = [t for t in cleaned if t not in STOP_TOKENS]
    # abbreviations
    cleaned = [ANIMAL_ABBREV.get(t, t) for t in cleaned]

    hits: List[str] = []
    seen: Set[str] = set()

    def add_if_canonical(phrase: str):
        key = phrase.lower()
        if key in CANONICAL_TAGS and key not in seen:
            seen.add(key)
            # pretty-case words
            hits.append(" ".join(w.capitalize() for w in phrase.split()))

    # tri-grams
    for i in range(len(cleaned) - 2):
        phrase = " ".join(cleaned[i:i+3])
        add_if_canonical(phrase)

    # bi-grams
    for i in range(len(cleaned) - 1):
        phrase = " ".join(cleaned[i:i+2])
        add_if_canonical(phrase)

    # singles
    for t in cleaned:
        add_if_canonical(t)

    return hits

def write_keywords_exif(path: Path, tags: List[str]) -> None:
    if not tags or path.suffix.lower() not in {'.jpg', '.jpeg', '.tif', '.tiff'} or piexif is None:
        return
    try:
        ex = piexif.load(str(path))
        keywords = "; ".join(sorted({t.strip() for t in tags if t.strip()}))
        ex['0th'][XPKeywords] = _encode_xp(keywords)
        piexif.insert(piexif.dump(ex), str(path))
        _debug(f"Wrote XPKeywords to {path.name}: {keywords}")
    except Exception as e:
        _warn(f"Failed writing XPKeywords for {path}: {e}")


# =========================
# Main
# =========================
def collect_unique_tags(items: List[Dict[str, object]]) -> List[str]:
    uniq: Dict[str, str] = {}
    for it in items:
        for t in it.get("Tags", []):
            k = t.lower()
            if k not in uniq:
                uniq[k] = t
    return [uniq[k] for k in sorted(uniq.keys())]

def organize_images(
    src_folder: Path,
    recursive: bool,
    site: str,
    include_seconds: bool,
    copyright_holder: str,
    site_info: str = ""
):
    """
    Headless core. Returns (log_csv_path, rows_written, unique_tags_in_run, errors).
    """
    _info(f"Starting Image Organizer in folder: {src_folder} (recursive={recursive})")
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {src_folder}")

    files = iter_images(src_folder, recursive)
    if not files:
        raise RuntimeError("No images found in the selected folder.")

    # read metadata pass
    item_infos = []
    for p in files:
        meta = read_image_metadata(p)
        dt, from_meta = choose_dt_for_filename(p, meta)
        if not from_meta:
            _warn(f"No timestamp in metadata for {p.name}; used file mtime.")
        # enrich tags from filename (exact canonical matches only)
        fn_tags = infer_exact_tags_from_filename(p, site)
        if fn_tags:
            existing_lower = {t.lower() for t in meta.get("Tags", [])}
            for t in fn_tags:
                if t.lower() not in existing_lower:
                    meta.setdefault("Tags", []).append(t)
                    existing_lower.add(t.lower())
        item_infos.append({"path": p, "meta": meta, "dt_for_name": dt})

    unique_tags = collect_unique_tags([i["meta"] for i in item_infos])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv = src_folder / f"rename_log_{ts}.csv"

    base_fields = [
        # identity
        "original_file_name", "new_file_name",
        # absolute paths to enable robust revert
        "original_full_path", "new_full_path",
        # clickable Excel links
        "original_link", "new_link",
        # time context
        "datetime_from_metadata", "converted_at",
        # metadata snapshot
        "Comments", "Title", "Subject", "Tags",
        "Copyright",
        "CopyrightExisting",
    ]
    header = base_fields + unique_tags

    used_names: Dict[str, int] = {}  # for collision management within this run (per dir)
    rows_out: List[Dict[str, str]] = []
    errors: List[str] = []

    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for info in item_infos:
            p = info["path"]; meta = info["meta"]; dt = info["dt_for_name"]
            ext = p.suffix

            # STRICT user choice: do not auto-include seconds
            new_name = make_filename(dt, site, include_seconds, ext)

            # Duplicate handling:
            # 1) resolution suffix, 2) numeric counter
            if new_name in used_names:
                w, h = get_image_size_safe(p)
                if w > 0 and h > 0:
                    cand = append_resolution_suffix(new_name, w, h)
                    if cand not in used_names:
                        new_name = cand
                        used_names[new_name] = 1
                    else:
                        ctr = used_names.get(cand, 1) + 1
                        new_name = append_counter_suffix(cand, ctr)
                        used_names[new_name] = ctr
                else:
                    ctr = used_names.get(new_name, 1) + 1
                    new_name = append_counter_suffix(new_name, ctr)
                    used_names[new_name] = ctr
            else:
                used_names[new_name] = 1

            try:
                new_path = safe_rename(p, new_name)

                # Comments <- existing + Title + Site Info; Title cleared
                final_comments = write_comments_and_clear_title(new_path, meta, site_info)

                # Copyright write (skip if exists)
                try:
                    write_copyright_exif(new_path, copyright_holder, skip_if_exists=True)
                except Exception as e:
                    _warn(f"EXIF Copyright write issue for {new_path}: {e}")

                # Write merged tags back to XPKeywords
                try:
                    write_keywords_exif(new_path, meta.get("Tags", []))
                except Exception as e:
                    _warn(f"Keyword write issue for {new_path}: {e}")

                converted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                original_full_path = str(p.resolve())
                new_full_path = str(new_path.resolve())

                row = {
                    "original_file_name": p.name,
                    "new_file_name": new_path.name,
                    "original_full_path": original_full_path,
                    "new_full_path": new_full_path,
                    "original_link": _excel_hyperlink(original_full_path, p.name),
                    "new_link": _excel_hyperlink(new_full_path, new_path.name),
                    "datetime_from_metadata": meta.get("DateTimeFromMetadata", ""),
                    "converted_at": converted_at,
                    "Comments": final_comments,
                    "Title": "",  # cleared
                    "Subject": str(meta.get("Subject","")),
                    "Tags": "; ".join(meta.get("Tags", [])),
                    "Copyright": copyright_holder or "",
                    "CopyrightExisting": str(meta.get("CopyrightExisting","")),
                }
                tags_lower = {t.lower() for t in meta.get("Tags", [])}
                for t in unique_tags:
                    row[t] = 1 if t.lower() in tags_lower else 0

                writer.writerow(row)
                rows_out.append(row)
                _info(f"Renamed {p.name} -> {new_path.name}")
            except Exception as e:
                msg = f"Rename failed for {p}: {e}"
                _err(msg); errors.append(msg)

    _info(f"Wrote CSV log: {log_csv}")
    return log_csv, rows_out, unique_tags, errors


def example_filename(site: str, include_seconds: bool, example_dt: Optional[datetime] = None, ext: str = ".jpg") -> str:
    """
    Example preview (UI): respects only the checkbox; does not auto-include seconds.
    """
    if example_dt is None: example_dt = datetime.now()
    return make_filename(example_dt, site, include_seconds, ext)


# =========================
# Revert
# =========================
def revert_operation(src_folder: Path, log_csv: Path) -> Tuple[int, List[str]]:
    """
    Revert renames using the CSV log: new_full_path -> original_full_path.
    Falls back to src_folder/new_file_name -> src_folder/original_file_name if abs paths absent.
    Avoids overwriting: if original exists, the 'new' file is moved to _revert_conflicts/.
    Returns: (reverted_count, error_messages)
    """
    reverted = 0
    errors: List[str] = []
    try:
        with log_csv.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            has_abs_cols = {"original_full_path", "new_full_path"}.issubset(reader.fieldnames or [])
            for row in reader:
                orig_name = row.get("original_file_name") or ""
                new_name  = row.get("new_file_name") or ""
                if not orig_name or not new_name:
                    continue

                # Prefer absolute paths if present
                if has_abs_cols:
                    orig_path = Path(row.get("original_full_path", "")).resolve()
                    new_path  = Path(row.get("new_full_path", "")).resolve()
                else:
                    orig_path = (src_folder / orig_name).resolve()
                    new_path  = (src_folder / new_name).resolve()

                try:
                    if not new_path.exists():
                        # try fallback when absolute path points nowhere
                        fallback_new = (src_folder / new_name).resolve()
                        if fallback_new.exists():
                            new_path = fallback_new
                        else:
                            errors.append(f"Missing file to revert: {new_path}")
                            continue

                    # If original already exists, don't overwrite—move 'new' into conflicts
                    if orig_path.exists():
                        conflict_dir = src_folder / "_revert_conflicts"
                        conflict_dir.mkdir(exist_ok=True)
                        target = conflict_dir / new_path.name
                        i = 1
                        while target.exists():
                            target = conflict_dir / f"{new_path.stem}_{i:02d}{new_path.suffix}"
                            i += 1
                        new_path.rename(target)
                        errors.append(f"Conflict: {orig_path.name} exists. Moved {new_path.name} -> {target.name}")
                        continue

                    # Ensure destination dir exists (handles nested originals)
                    orig_path.parent.mkdir(parents=True, exist_ok=True)
                    new_path.rename(orig_path)
                    reverted += 1

                except Exception as e:
                    errors.append(f"Failed revert {new_name} -> {orig_name}: {e}")

    except Exception as e:
        errors.append(f"Failed to read log CSV: {e}")

    return reverted, errors
