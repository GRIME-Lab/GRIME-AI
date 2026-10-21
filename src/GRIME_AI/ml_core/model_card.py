#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
Model card writer for models fine-tuned in GRIME AI.

Layout follows the Hugging Face model card format (YAML metadata header +
Markdown body), using the facebook/sam2.1-hiera-large card as the template.
To publish on Hugging Face, rename the file to README.md.

Every value comes from the training run. Fields the run cannot know
(intended use, authors, license of the source imagery) are written as
clearly marked placeholders for the user to complete.
"""

import os
import re
from collections import Counter
from datetime import datetime

# Base checkpoint file -> Hugging Face model id (lookup table, not tuning values)
BASE_MODEL_IDS = {
    "sam2.1_hiera_large.pt": "facebook/sam2.1-hiera-large",
    "sam2.1_hiera_base_plus.pt": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_hiera_small.pt": "facebook/sam2.1-hiera-small",
    "sam2.1_hiera_tiny.pt": "facebook/sam2.1-hiera-tiny",
}

SAM2_CITATION = """@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}"""

TODO = "_[To be completed by the model author]_"

# Keys saved in GRIME AI SAM 2 checkpoints (sam2_trainer._save_model_checkpoint
# and _update_checkpoints_blob_radius). Keep in step with the trainer.
CHECKPOINT_KEYS = [
    ("model_state_dict", "Fine-tuned SAM 2 weights (full model state dict)"),
    ("base_model", "Base model family (sam2)"),
    ("base_checkpoint", "Base checkpoint file the fine-tune started from"),
    ("site_name", "Site the model was trained for"),
    ("target_category_name", "Category the model segments"),
    ("categories", "All annotation categories (id and name)"),
    ("num_classes", "Number of annotation categories"),
    ("creation_UTC", "Time the checkpoint was saved (UTC)"),
    ("learning_rate", "Learning rate of the run"),
    ("epochs", "Epoch at which this checkpoint was saved"),
    ("selection_rule", "How checkpoints are ranked (lowest validation loss)"),
    ("val_loss", "Validation loss at this epoch"),
    ("val_accuracy", "Validation pixel accuracy at this epoch (includes background)"),
    ("pooled_target_iou", "Pooled IoU of the target class over sampled validation pixels"),
    ("miou", "Same value as pooled_target_iou (older name, kept for compatibility)"),
    ("training_split_manifest", "File name of the split manifest listing every training and validation image"),
    ("prompt_protocol", "Prompt mode, maximum positive prompts and negative balance used in training; inference reproduces it"),
    ("category_centroids", "Per-category centroids of annotated regions (pixel and normalized), used to build inference prompts"),
    ("blob_filter_radius", "Circular blob-filter radius as a fraction of the image diagonal"),
    ("blob_filter_mahal_fraction", "Fallback radius fraction from the centroid distribution (present after the blob-filter update)"),
    ("blob_centroid_mean", "Mean of training centroids for the Mahalanobis blob filter (present after the blob-filter update)"),
    ("blob_centroid_cov", "2x2 covariance of training centroids (present after the blob-filter update)"),
    ("blob_filter_n_sigma", "Mahalanobis distance threshold (present after the blob-filter update)"),
    ("blob_centroids", "All training centroids for the kNN blob filter (present after the blob-filter update)"),
    ("blob_knn_k", "k for the kNN blob filter (present after the blob-filter update)"),
    ("blob_knn_threshold", "kNN distance threshold in pixels (present after the blob-filter update)"),
    ("suggested_blob_filter_mode", "Recommended blob-filter mode; advisory only (present after the blob-filter update)"),
]

# NEON / PhenoCam style: ..._YYYY_MM_DD_HHMMSS.jpg
_TS_PATTERNS = [
    re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})"),
    re.compile(r"(\d{4})(\d{2})(\d{2})[_T-]?(\d{2})(\d{2})(\d{2})"),
]


def _parse_timestamp(name):
    for pat in _TS_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return datetime(*[int(g) for g in m.groups()])
            except ValueError:
                continue
    return None


def _fmt(v, spec="{:.4f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "N/A"


def _coverage(entries):
    """Date range, months and hours of day covered by a list of manifest entries."""
    stamps = [t for t in (_parse_timestamp(e.get("file_name", "")) for e in entries) if t]
    if not stamps:
        return None
    months = Counter(t.month for t in sorted(stamps))
    months = {datetime(2000, m, 1).strftime("%b"): months[m] for m in sorted(months)}
    hours = sorted({t.strftime("%H:%M") for t in stamps})
    return {
        "first": min(stamps).strftime("%Y-%m-%d"),
        "last": max(stamps).strftime("%Y-%m-%d"),
        "months": months,
        "hours": hours,
        "parsed": len(stamps),
        "total": len(entries),
    }


def _metrics_table(best, final):
    rows = [
        ("Dice (target class)", "dice"),
        ("IoU (target class)", "iou"),
        ("Precision (target class)", "precision"),
        ("Recall (target class)", "recall"),
        ("Average precision (area under PR curve)", "average_precision"),
        ("Validation loss", "val_loss"),
        ("Pixel accuracy (includes background)", "pixel_accuracy"),
    ]
    head = ["| Metric | Saved model (best checkpoint) | Final epoch |",
            "|---|---|---|"]
    body = [f"| {label} | {_fmt((best or {}).get(key))} | {_fmt((final or {}).get(key))} |"
            for label, key in rows]
    return "\n".join(head + body)


def build_model_card(info):
    site = info.get("site_name") or "unknown_site"
    target = info.get("target_label") or "target"
    base_ckpt = info.get("base_checkpoint")
    base_id = BASE_MODEL_IDS.get(base_ckpt or "")
    best = info.get("best_eval") or {}
    final = info.get("final_epoch") or {}
    manifest = info.get("split_manifest") or {}
    fg = info.get("foreground_stats") or {}
    cats = info.get("categories") or []

    train_entries = manifest.get("train", [])
    val_entries = manifest.get("val", [])
    cov_train = _coverage(train_entries)
    cov_all = _coverage(train_entries + val_entries)

    # ---------------- YAML metadata ----------------
    yaml = ["---", "license: apache-2.0", "pipeline_tag: mask-generation",
            "library_name: pytorch"]
    if base_id:
        yaml += [f"base_model: {base_id}", "base_model_relation: finetune"]
    yaml += ["tags:", "  - grime-ai", "  - sam2", "  - image-segmentation",
             "  - environmental-monitoring", "  - time-lapse-imagery",
             f"  - {re.sub(r'[^a-z0-9-]+', '-', target.lower()).strip('-') or 'target'}",
             "metrics:", "  - dice", "  - iou", "  - precision", "  - recall", "---"]

    out = ["\n".join(yaml), ""]

    # ---------------- Title / intro (mirrors the SAM 2.1 card) ----------------
    out += [f"# GRIME AI SAM 2 fine-tune: {target} ({site})", ""]
    intro = (f"Segmentation model for **{target}** in fixed-camera imagery from site **{site}**, "
             f"fine-tuned in GRIME AI (GaugeCam Remote Image Manager Educational, Artificial Intelligence)")
    intro += f" from [{base_id}](https://huggingface.co/{base_id})." if base_id else "."
    out += [intro, "",
            "The base model is described in the [SAM 2 paper](https://arxiv.org/abs/2408.00714); "
            "its official code is at https://github.com/facebookresearch/sam2.", ""]

    # ---------------- Model details ----------------
    pp = info.get("prompt_protocol") or {}
    cat_text = ", ".join(f"{c.get('name')} (id {c.get('id')})" for c in cats) or "N/A"
    out += ["## Model details", "",
            "| Field | Value |", "|---|---|",
            f"| Base model | {base_id or base_ckpt or 'N/A'} |",
            f"| Fine-tuned modules | Prompt encoder and mask decoder (image encoder frozen; embeddings cached) |",
            f"| Target category | {target} |",
            f"| Annotation categories | {cat_text} |",
            f"| Prompt protocol | train: {pp.get('train')}, validation: {pp.get('val')}; "
            f"max positives {pp.get('max_positives')}, negative balance {pp.get('negative_balance')}:1 |",
            f"| Blob filter radius | {_fmt(info.get('blob_filter_radius_pct'), '{:.1f}')}% of image diagonal |",
            f"| Created | {info.get('formatted_time', 'N/A')} |",
            f"| Developed by | {TODO} |",
            f"| License | Apache 2.0 (model weights and code). Source imagery license: {TODO} |",
            ""]

    # ---------------- Usage (mirrors the SAM 2.1 card) ----------------
    ckpts = info.get("checkpoint_files") or []
    best_file = best.get("checkpoint_file") or (ckpts[0] if ckpts else "<checkpoint>.torch")
    out += ["## Usage", "",
            "Load the checkpoint for inference in GRIME AI. Inference reads the prompt protocol and "
            "blob-filter settings stored in the checkpoint, so they match training.", "",
            "To list everything stored in the checkpoint outside GRIME AI:", "",
            "```python",
            "import torch",
            "",
            f'ckpt = torch.load("{best_file}", map_location="cpu", weights_only=False)',
            "",
            "for key, value in ckpt.items():",
            '    if key == "model_state_dict":',
            "        n_params = sum(t.numel() for t in value.values())",
            '        print(f"{key}: {len(value)} tensors, {n_params:,} parameters")',
            '    elif key == "category_centroids":',
            "        for cat_id, points in value.items():",
            '            print(f"{key}[{cat_id}]: {len(points)} centroids")',
            "            for p in points:",
            '                print(f"    pixel {p[\'centroid_px\']}  normalized {p[\'centroid_norm\']}")',
            '    elif key == "blob_centroids" and value is not None:',
            '        print(f"{key}: {len(value)} points")',
            "        for p in value:",
            '            print(f"    {p}")',
            "    else:",
            '        print(f"{key}: {value}")',
            "",
            "# Fine-tuned SAM 2 weights, ready for model.load_state_dict()",
            'state_dict = ckpt["model_state_dict"]',
            "```", "",
            "### Checkpoint contents", "",
            "Keys written by the current GRIME AI trainer. Checkpoints from older builds may lack some.", "",
            "| Key | Contents |", "|---|---|",
        ] + [f"| `{k}` | {d} |" for k, d in CHECKPOINT_KEYS] + [""]

    # ---------------- Training data ----------------
    out += ["## Training data", ""]
    out += [f"- Training images: {len(train_entries) or info.get('num_train_images', 'N/A')}",
            f"- Validation images: {len(val_entries) or info.get('num_val_images', 'N/A')}"]
    ss = manifest.get("split_settings") or {}
    if ss:
        out.append(f"- Split: train {ss.get('train_split')}, validation {ss.get('val_split')}"
                   + (f"; seasonal hold-out {', '.join(ss['holdout_seasons'])} ({ss.get('season_type')})"
                      if ss.get("holdout_seasons") else ""))
    if cov_all:
        out.append(f"- Date range: {cov_all['first']} to {cov_all['last']}")
        out.append("- Months represented: " + ", ".join(f"{m} ({n})" for m, n in cov_all["months"].items()))
        out.append("- Times of day represented (from image file names): " + ", ".join(cov_all["hours"]))
    for split in ("train", "val"):
        st = fg.get(split) or {}
        if st.get("count"):
            out.append(f"- Target pixel fraction ({split}): mean {st['mean']:.4f}, "
                       f"median {st['median']:.4f}, std {st['std']:.4f}")
    if info.get("split_manifest_file"):
        out.append(f"- Exact images: `{info['split_manifest_file']}` lists every training and "
                   f"validation image by root-relative path and SHA-256 hash, plus the annotation "
                   f"file hashes.")
    out.append("")

    # ---------------- Training procedure ----------------
    out += ["## Training procedure", "",
            "| Hyperparameter | Value |", "|---|---|",
            f"| Optimizer | {info.get('optimizer', 'N/A')} |",
            f"| Loss | {info.get('loss_function', 'N/A')} |",
            f"| Learning rate | {info.get('learning_rate')} |",
            f"| Weight decay | {info.get('weight_decay')} |",
            f"| Batch size | {info.get('batch_size')} |",
            f"| Epochs configured / completed | {info.get('epochs_configured')} / {info.get('epochs_completed')} |",
            f"| Early stopping | {'triggered at epoch ' + str(info.get('early_stop_epoch')) if info.get('early_stopped') else 'not triggered'} |",
            f"| LR scheduler | {'ReduceLROnPlateau' if info.get('lr_scheduler_enabled') else 'none'} |",
            f"| Checkpoint selection | {best.get('selection_rule', 'lowest validation loss')} |",
            f"| Device | {info.get('device', 'N/A')} |",
            ""]

    # ---------------- Evaluation ----------------
    out += ["## Evaluation", "",
            "These are **validation** results. The validation images also chose the saved "
            "checkpoint, so the numbers are somewhat optimistic. Performance on images the "
            "user tests at inference time should be reported separately.", ""]
    if best.get("epoch") is not None:
        line = f"Saved model: epoch {best.get('epoch')}"
        if best.get("checkpoint_file"):
            line += f", file `{best['checkpoint_file']}`"
        out += [line + ".", ""]
    out += [_metrics_table(best, final), "",
            "Dice, IoU, precision and recall are computed on the target class only. Pixel "
            "accuracy counts background pixels and stays near 1.0 for small targets regardless "
            "of segmentation quality, so it is listed for reference only.", ""]
    if best.get("target_pixel_fraction") is not None:
        out += [f"No-skill average precision equals the target pixel fraction, "
                f"{best['target_pixel_fraction']:.4f}.", ""]

    # ---------------- Limitations ----------------
    out += ["## Limitations", ""]
    lim = [f"Trained and validated at one site ({site}); performance at other sites is untested."]
    if cov_train and len(cov_train["hours"]) <= 2:
        lim.append(f"Training images cover only {', '.join(cov_train['hours'])} (time in the image file names); "
                   f"other lighting conditions (early morning, dusk, night/IR) are not represented.")
    if cov_train:
        missing = [m for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                               "Sep", "Oct", "Nov", "Dec") if m not in cov_train["months"]]
        if missing:
            lim.append(f"Months absent from training: {', '.join(missing)}.")
    nv = len(val_entries) or best.get("n_val_images")
    if nv:
        lim.append(f"Validation uses {nv} image(s); small differences between runs are within noise.")
    lim.append(f"Out-of-scope uses: {TODO}")
    out += [f"- {x}" for x in lim] + [""]

    # ---------------- Files ----------------
    out += ["## Files in the model folder", "",
            "- `*.torch`: ranked best checkpoints (lowest validation loss first)",
            "- `*_configuration.txt`: run configuration",
            "- `*_Training_Report_*.pdf`: diagnostic report and graphs",
            "- `graphs/`: per-epoch metric plots and `*_metrics_*.csv`",
            "- `validation_overlays/`: predictions on validation images (`best_ep###` = saved model)",
            "- `diagnostics/`: split manifest and centroid diagnostic", ""]

    # ---------------- Citation ----------------
    out += ["## Citation", "",
            f"GRIME AI citation: {TODO}", "",
            "Base model:", "", "```", SAM2_CITATION, "```", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# PDF rendering of the Markdown card (fpdf2, already a GRIME AI dependency).
# Layout values only; callers may pass their own style dict.
# All text is black for readability.
# ----------------------------------------------------------------------------
PDF_STYLE = {
    "margin_mm": 15,
    "font": "Helvetica",
    "mono_font": "Courier",
    "title_size": 16,
    "section_size": 12,
    "body_size": 10,
    "table_size": 9,
    "code_size": 8,
    "line_h": 5.5,
    "code_line_h": 4.2,
    "section_fill": (230, 230, 230),
    "code_fill": (245, 245, 245),
    "note_fill": (250, 244, 225),
    "text_rgb": (0, 0, 0),
    "bullet_indent_mm": 5,
}

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TODO_RE = re.compile(r"(?<![A-Za-z0-9])_(\[[^\]]*\])_(?![A-Za-z0-9])")


def _pdf_safe(text):
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _inline(text, keep_bold=True):
    """Markdown inline -> fpdf2 markdown (**bold**, __italic__); links as 'text (url)'."""
    text = _LINK_RE.sub(lambda m: m.group(1) if m.group(1) == m.group(2)
                        else f"{m.group(1)} ({m.group(2)})", text)
    text = _TODO_RE.sub(r"__\1__", text)
    text = text.replace("`", "")
    if not keep_bold:
        text = text.replace("**", "").replace("__", "")
    return _pdf_safe(text)


def _split_front_matter(md):
    lines = md.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return lines[1:i], lines[i + 1:]
    return [], lines


def _front_matter_rows(fm_lines):
    rows, key = [], None
    for ln in fm_lines:
        if ln.startswith("  - ") and key is not None:
            rows[-1][1] = (rows[-1][1] + ", " if rows[-1][1] else "") + ln[4:].strip()
        elif ":" in ln:
            key, val = ln.split(":", 1)
            rows.append([key.strip(), val.strip()])
    return rows


def write_model_card_pdf(md_text, pdf_path, footer_text="", style=None):
    from fpdf import FPDF

    st = dict(PDF_STYLE, **(style or {}))
    pdf = FPDF()
    pdf.set_margins(st["margin_mm"], st["margin_mm"], st["margin_mm"])
    pdf.set_auto_page_break(auto=True, margin=st["margin_mm"])
    if footer_text:
        _foot = _pdf_safe(footer_text)

        def _footer():
            pdf.set_y(-st["margin_mm"])
            pdf.set_font(st["font"], "I", st["code_size"])
            pdf.set_text_color(*st["text_rgb"])
            pdf.cell(0, st["code_line_h"], f"{_foot}  |  page {pdf.page_no()}", align="C")
        pdf.footer = _footer
    pdf.add_page()
    pdf.set_text_color(*st["text_rgb"])
    page_w = pdf.w - 2 * st["margin_mm"]
    lh = st["line_h"]

    def para(text, size=None, style_flag=""):
        pdf.set_font(st["font"], style_flag, size or st["body_size"])
        pdf.set_x(st["margin_mm"])
        pdf.multi_cell(page_w, lh, _inline(text), markdown=True)
        pdf.ln(1.5)

    def section(text):
        pdf.ln(2)
        pdf.set_font(st["font"], "B", st["section_size"])
        pdf.set_fill_color(*st["section_fill"])
        pdf.set_x(st["margin_mm"])
        pdf.cell(page_w, lh + 2, _inline(text, keep_bold=False), fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def table(rows):
        rows = [[_inline(c, keep_bold=False) for c in r] for r in rows]
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        pdf.set_font(st["font"], "", st["table_size"])
        if hasattr(pdf, "table"):
            widths = (35, 65) if ncol == 2 else None
            with pdf.table(col_widths=widths, line_height=lh - 0.5,
                           text_align="LEFT", first_row_as_headings=True) as t:
                for r in rows:
                    tr = t.row()
                    for c in r:
                        tr.cell(c)
        else:  # older fpdf2: one "Header: value" line per cell
            head = rows[0]
            for r in rows[1:]:
                pdf.set_x(st["margin_mm"])
                pdf.multi_cell(page_w, lh, "  |  ".join(
                    f"{h}: {c}" if i else c for i, (h, c) in enumerate(zip(head, r))))
        pdf.ln(2)

    def code(lines):
        pdf.set_font(st["mono_font"], "", st["code_size"])
        pdf.set_fill_color(*st["code_fill"])
        for ln in lines:
            pdf.set_x(st["margin_mm"])
            pdf.multi_cell(page_w, st["code_line_h"], _pdf_safe(ln) or " ", fill=True)
        pdf.ln(2)

    def note(text):
        pdf.set_font(st["font"], "", st["body_size"])
        pdf.set_fill_color(*st["note_fill"])
        pdf.set_x(st["margin_mm"])
        pdf.multi_cell(page_w, lh, _inline(text), markdown=True, fill=True)
        pdf.ln(2)

    fm, body = _split_front_matter(md_text)
    meta_rows = _front_matter_rows(fm)

    i, n = 0, len(body)
    title_done = False
    while i < n:
        ln = body[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            pdf.set_font(st["font"], "B", st["title_size"])
            pdf.set_x(st["margin_mm"])
            pdf.multi_cell(page_w, lh + 2, _inline(s[2:], keep_bold=False))
            pdf.ln(2)
            title_done = True
            i += 1
            continue
        if s.startswith("### "):
            pdf.ln(1)
            pdf.set_font(st["font"], "B", st["body_size"] + 1)
            pdf.set_x(st["margin_mm"])
            pdf.multi_cell(page_w, lh, _inline(s[4:], keep_bold=False))
            pdf.ln(1)
            i += 1
            continue
        if s.startswith("## "):
            if title_done and meta_rows:   # metadata right after the intro
                section("Metadata")
                table([["Field", "Value"]] + meta_rows)
                meta_rows = []
            section(s[3:])
            i += 1
            continue
        if s.startswith("```"):
            j, block = i + 1, []
            while j < n and not body[j].strip().startswith("```"):
                block.append(body[j])
                j += 1
            code(block)
            i = j + 1
            continue
        if s.startswith("|"):
            rows = []
            while i < n and body[i].strip().startswith("|"):
                cells = [c.strip() for c in body[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table(rows)
            continue
        if s.startswith(">"):
            buf = []
            while i < n and body[i].strip().startswith(">"):
                buf.append(body[i].strip()[1:].strip())
                i += 1
            note(" ".join(buf))
            continue
        if s.startswith("- "):
            pdf.set_font(st["font"], "", st["body_size"])
            ind = st["bullet_indent_mm"]
            pdf.set_x(st["margin_mm"])
            pdf.cell(ind, lh, "-")
            pdf.multi_cell(page_w - ind, lh, _inline(s[2:]), markdown=True)
            pdf.ln(0.5)
            i += 1
            continue
        buf = []
        while i < n and body[i].strip() and not re.match(r"(#|\||```|>|- )", body[i].strip()):
            buf.append(body[i].strip())
            i += 1
        para(" ".join(buf))

    pdf.output(pdf_path)
    print(f"[model_card] written: {pdf_path}")
    return pdf_path


def write_model_card(path, info, write_pdf=True, style=None):
    """Write the Markdown card to `path` and, by default, a PDF beside it."""
    text = build_model_card(info)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[model_card] written: {path}")
    if write_pdf:
        pdf_path = os.path.splitext(path)[0] + ".pdf"
        footer = (f"GRIME AI model card  |  {info.get('formatted_time', '')}  |  "
                  f"Site: {info.get('site_name', '')}")
        try:
            write_model_card_pdf(text, pdf_path, footer_text=footer, style=style)
        except Exception as e:
            print(f"[model_card] PDF not written: {e}")
    return path
