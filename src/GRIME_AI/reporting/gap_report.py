"""
GRIME_AI_gap_report
===================
Generate a self-contained HTML + CSV report of temporal discontinuities in a
folder of downloaded time-lapse images (USGS HIVIS, PhenoCam, NEON).

Design:
  * Timestamps are parsed from filenames via a pattern list (extensible).
  * Expected cadence is inferred (mode of consecutive deltas) unless supplied.
  * If imagery occupies only part of each day (e.g. 11:00-13:00), the daily
    active window is detected and gaps are judged against that window only.
  * Output is dependency-free HTML (inline CSS + SVG heatmap) plus a CSV,
    written into the download folder itself.

Usage:
    from GRIME_AI_gap_report import generate_gap_report
    generate_gap_report(download_folder)                     # after a fetch
    generate_gap_report(download_folder, tz="America/Chicago")

CLI:
    python GRIME_AI_gap_report.py <folder-or-listing.txt> [--tz ZONE]
"""

from __future__ import annotations

import csv
import html
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# (regex, is_utc). First match wins. Extend as new networks are added.
FILENAME_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    # USGS HIVIS / ISO-like:  ..._2026-01-01T17-00-31Z.jpg
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z"), True),
    # ISO with colons:        ..._2026-01-01T17:00:31Z...
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z"), True),
    # PhenoCam / NEON:        sitename_2026_01_01_170031.jpg  (site-local)
    (re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})"), False),
    # Compact:                ..._20260101_170031...          (assume local)
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_T-](\d{2})(\d{2})(\d{2})"), False),
]



# USGS /cameras "tz" strings are often bare abbreviations. Map them to IANA
# zones so gap analysis is DST-aware (a fixed offset would make the capture
# schedule appear to shift an hour every March and November).
USGS_TZ_TO_IANA = {
    "EST": "America/New_York", "EDT": "America/New_York",
    "CST": "America/Chicago",  "CDT": "America/Chicago",
    "MST": "America/Denver",   "MDT": "America/Denver",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
    "AKST": "America/Anchorage", "AKDT": "America/Anchorage",
    "HST": "Pacific/Honolulu", "AST": "America/Puerto_Rico",
    "US/EASTERN": "America/New_York", "US/CENTRAL": "America/Chicago",
    "US/MOUNTAIN": "America/Denver", "US/PACIFIC": "America/Los_Angeles",
    "US/ALASKA": "America/Anchorage", "US/HAWAII": "Pacific/Honolulu",
    "UTC": "UTC",
}


def resolve_usgs_tz(tz_str):
    """Best-effort: USGS camera 'tz' string -> IANA zone name (or None)."""
    if not tz_str:
        return None
    key = tz_str.upper().strip().replace(" ", "_")
    if key in USGS_TZ_TO_IANA:
        return USGS_TZ_TO_IANA[key]
    if ZoneInfo is not None:
        try:
            ZoneInfo(tz_str)
            return tz_str          # already a valid IANA name
        except Exception:
            return None
    return None


def parse_timestamp(name: str) -> Optional[Tuple[datetime, bool]]:
    """Return (naive datetime, is_utc) parsed from a filename, or None."""
    for pat, is_utc in FILENAME_PATTERNS:
        m = pat.search(name)
        if m:
            y, mo, d, h, mi, s = (int(g) for g in m.groups())
            try:
                return datetime(y, mo, d, h, mi, s), is_utc
            except ValueError:
                continue
    return None


def _collect(source) -> List[str]:
    """Accept a folder of images or a text listing; return filenames."""
    p = Path(source)
    if p.is_dir():
        return [f.name for f in sorted(p.iterdir())
                if f.suffix.lower() in IMAGE_EXTS]
    if p.is_file():
        return [line.strip() for line in
                p.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()]
    raise FileNotFoundError(source)


def _localize(ts: datetime, is_utc: bool, tz) -> datetime:
    if is_utc:
        aware = ts.replace(tzinfo=timezone.utc)
        return aware.astimezone(tz) if tz else aware
    return ts  # already site-local; leave naive


def _infer_cadence_minutes(times: List[datetime]) -> int:
    deltas = Counter()
    for a, b in zip(times, times[1:]):
        d = round((b - a).total_seconds() / 60)
        if 1 <= d <= 24 * 60:
            deltas[d] += 1
    if not deltas:
        return 15
    return deltas.most_common(1)[0][0]


def _daily_window(by_day: dict, cadence: int) -> Tuple[int, int]:
    """Typical (start_min, end_min) of daily coverage, minute-of-day.
    Uses medians so single anomalous days don't stretch the window."""
    starts = sorted(min(v) for v in by_day.values())
    ends = sorted(max(v) for v in by_day.values())
    start = starts[len(starts) // 2]
    end = ends[len(ends) // 2]
    if (end - start) >= 24 * 60 - cadence:  # effectively continuous
        return 0, 24 * 60 - cadence
    return start, end


def analyze(source, tz_name: Optional[str] = None,
            cadence_minutes: Optional[int] = None) -> dict:
    tz = ZoneInfo(tz_name) if (tz_name and ZoneInfo) else None
    parsed = []
    n_files = 0
    for name in _collect(source):
        n_files += 1
        r = parse_timestamp(name)
        if r:
            parsed.append(_localize(r[0], r[1], tz))
    parsed.sort()
    if not parsed:
        raise ValueError("No parseable timestamps found in %s" % source)

    cadence = cadence_minutes or _infer_cadence_minutes(parsed)

    # Snap each capture to its nearest cadence slot (absorbs clock jitter).
    by_day: dict = defaultdict(set)
    for ts in parsed:
        mod = ts.hour * 60 + ts.minute + (1 if ts.second >= 30 else 0)
        by_day[ts.date()].add(round(mod / cadence) * cadence)

    w0, w1 = _daily_window(by_day, cadence)
    slots = list(range(w0, w1 + 1, cadence))

    first, last = min(by_day), max(by_day)
    all_days = [first + timedelta(days=i)
                for i in range((last - first).days + 1)]

    day_rows = []       # (date, present_set, missing_list)
    gap_events = []     # contiguous runs of missing slots
    open_gap = None     # (start_date, start_slot, count)

    for d in all_days:
        present = by_day.get(d, set())
        missing = [s for s in slots if s not in present]
        day_rows.append((d, present, missing))
        for s in slots:
            if s not in present:
                if open_gap is None:
                    open_gap = [d, s, d, s, 1]
                else:
                    open_gap[2], open_gap[3] = d, s
                    open_gap[4] += 1
            else:
                if open_gap:
                    gap_events.append(tuple(open_gap))
                    open_gap = None
    if open_gap:
        gap_events.append(tuple(open_gap))

    return {
        "source": str(source), "tz": tz_name or "as-parsed (UTC files kept UTC)",
        "n_files": n_files, "n_parsed": len(parsed),
        "cadence": cadence, "window": (w0, w1), "slots": slots,
        "first": first, "last": last, "day_rows": day_rows,
        "gap_events": gap_events,
        "missing_days": [d for d, p, _ in day_rows if not p],
        "n_missing_slots": sum(len(m) for _, _, m in day_rows),
        "n_expected": len(all_days) * len(slots),
    }


def _fmt(minute_of_day: int) -> str:
    return "%02d:%02d" % divmod(minute_of_day, 60)


def _svg_heatmap(a: dict) -> str:
    rows = a["day_rows"]
    slots = a["slots"]
    cw, ch, mx, my = 4, 16, 46, 4
    W = mx + len(rows) * cw + 4
    H = my + len(slots) * ch + 22
    out = ['<svg viewBox="0 0 %d %d" width="100%%" '
           'style="max-width:%dpx" role="img" '
           'aria-label="Image availability heatmap">' % (W, H, W)]
    for i, s in enumerate(slots):
        out.append('<text x="%d" y="%d" text-anchor="end" font-size="10" '
                   'fill="#777" font-family="sans-serif">%s</text>'
                   % (mx - 5, my + i * ch + ch // 2 + 3, _fmt(s)))
    last_month = None
    for j, (d, present, _) in enumerate(rows):
        if d.strftime("%Y-%m") != last_month:
            last_month = d.strftime("%Y-%m")
            x = mx + j * cw
            out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#bbb" '
                       'stroke-width="1"/>' % (x, my, x, my + len(slots) * ch))
            out.append('<text x="%d" y="%d" font-size="10" fill="#777" '
                       'font-family="sans-serif">%s</text>'
                       % (x, my + len(slots) * ch + 14, d.strftime("%b %Y")))
        for i, s in enumerate(slots):
            ok = s in present
            out.append('<rect x="%g" y="%d" width="%g" height="%d" rx="0.5" '
                       'fill="%s" fill-opacity="%s"><title>%s %s %s</title></rect>'
                       % (mx + j * cw, my + i * ch, cw - 0.6, ch - 2,
                          "#1D9E75" if ok else "#D64541",
                          "0.35" if ok else "1",
                          d.isoformat(), _fmt(s),
                          "present" if ok else "MISSING"))
    out.append("</svg>")
    return "".join(out)



def _svg_month_bars(a: dict) -> str:
    """Missing-images-per-month bar chart, pure SVG, no JS."""
    per_month = {}
    for d, _p, missing in a["day_rows"]:
        k = d.strftime("%Y-%m")
        per_month[k] = per_month.get(k, 0) + len(missing)
    keys = sorted(per_month)
    if not keys:
        return ""
    vmax = max(per_month.values()) or 1
    bw, gap, mx, my, bh = 46, 14, 34, 8, 110
    W = mx + len(keys) * (bw + gap) + 8
    H = my + bh + 34
    out = ['<svg viewBox="0 0 %d %d" width="%d" role="img" '
           'aria-label="Missing images per month">' % (W, H, W)]
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#ccc"/>'
               % (mx, my + bh, W - 4, my + bh))
    for i, k in enumerate(keys):
        v = per_month[k]
        h = round(bh * v / vmax)
        x = mx + i * (bw + gap)
        y = my + bh - h
        out.append('<rect x="%d" y="%d" width="%d" height="%d" rx="3" '
                   'fill="#D64541"><title>%s: %d missing</title></rect>'
                   % (x, y, bw, h, k, v))
        out.append('<text x="%d" y="%d" text-anchor="middle" font-size="11" '
                   'fill="#555" font-family="sans-serif">%d</text>'
                   % (x + bw // 2, y - 4, v))
        mon = datetime.strptime(k, "%Y-%m").strftime("%b")
        out.append('<text x="%d" y="%d" text-anchor="middle" font-size="11" '
                   'fill="#777" font-family="sans-serif">%s</text>'
                   % (x + bw // 2, my + bh + 15, mon))
    out.append("</svg>")
    return "".join(out)


def write_report(a: dict, out_dir) -> Tuple[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "image_gaps.csv"
    html_path = out_dir / "image_download_report.html"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap_start_date", "gap_start_time", "gap_end_date",
                    "gap_end_time", "n_missing_slots"])
        for d0, s0, d1, s1, n in a["gap_events"]:
            w.writerow([d0, _fmt(s0), d1, _fmt(s1), n])

    pct = 100.0 * (a["n_expected"] - a["n_missing_slots"]) / a["n_expected"]
    ev = sorted(a["gap_events"], key=lambda g: -g[4])
    ev_rows = "".join(
        "<tr><td>%s %s</td><td>%s %s</td><td style='text-align:right'>%d</td></tr>"
        % (d0, _fmt(s0), d1, _fmt(s1), n) for d0, s0, d1, s1, n in ev)

    doc = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Image download report</title><style>
body{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;max-width:1100px;margin:2rem auto;color:#1f1f1f;padding:0 1rem}
h1{font-size:22px;font-weight:600}h2{font-size:15px;font-weight:600;margin:2rem 0 .5rem;color:#333}
.sub{color:#666;font-size:13px;margin-top:-6px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:1.2rem 0}
.tile{background:#f6f6f4;border-radius:8px;padding:12px 18px;min-width:120px}
.tile .l{font-size:12px;color:#777}.tile .v{font-size:24px;font-weight:600;margin-top:2px}
.tile .v.bad{color:#B0332F}.tile .v.good{color:#0F6E56}
table.g{border-collapse:collapse;font-size:13px}
table.g td,table.g th{border:1px solid #e2e2e2;padding:4px 12px;text-align:left}
table.g th{background:#f6f6f4;font-weight:600}
table.g td:last-child{text-align:right}
.leg{font-size:13px;color:#555}
.leg span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 5px -1px 14px}
footer{color:#999;font-size:12px;margin:2.5rem 0 1rem}
</style></head><body>
<h1>Image download completeness report</h1>
<p class="sub">%(src)s &mdash; %(first)s to %(last)s &mdash; %(cad)d-min cadence,
daily window %(w0)s&ndash;%(w1)s (%(tz)s)</p>
<div class="tiles">
<div class="tile"><div class="l">Complete</div><div class="v %(pcls)s">%(pct).1f%%</div></div>
<div class="tile"><div class="l">Images on disk</div><div class="v">%(np)d</div></div>
<div class="tile"><div class="l">Missing</div><div class="v bad">%(miss)d</div></div>
<div class="tile"><div class="l">Gap events</div><div class="v">%(nev)d</div></div>
<div class="tile"><div class="l">Days fully absent</div><div class="v">%(ndays)d</div></div>
</div>
<h2>Availability heatmap</h2>
<p class="leg"><span style="background:#1D9E75;opacity:.45"></span>present
<span style="background:#D64541"></span>missing &mdash; hover any cell for date and time</p>
%(svg)s
<h2>Missing images per month</h2>
%(bars)s
<h2>Gap events (largest first)</h2>
<table class="g"><tr><th>From</th><th>Through</th><th>Missing images</th></tr>
%(ev)s</table>
<footer>Generated by GRIME AI &mdash; %(now)s. Machine-readable copy: image_gaps.csv</footer>
</body></html>""" % dict(
        src=html.escape(a["source"]), nf=a["n_files"], np=a["n_parsed"],
        first=a["first"], last=a["last"], cad=a["cadence"],
        w0=_fmt(a["window"][0]), w1=_fmt(a["window"][1]),
        tz=html.escape(a["tz"]), pct=pct, pcls="good" if pct >= 95 else "bad",
        miss=a["n_missing_slots"], exp=a["n_expected"],
        nev=len(a["gap_events"]), ndays=len(a["missing_days"]),
        svg=_svg_heatmap(a), bars=_svg_month_bars(a), ev=ev_rows,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"))

    html_path.write_text(doc, encoding="utf-8")
    return str(html_path), str(csv_path)



# ----------------------------------------------------------------------------
# PDF report (optional; requires matplotlib). Combines the summary tiles,
# heatmap, monthly bar chart, and the full gap-event table in one document.
# ----------------------------------------------------------------------------
def write_pdf_report(a: dict, out_dir) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return None

    out_path = str(Path(out_dir) / "image_download_report.pdf")
    rows = a["day_rows"]
    slots = a["slots"]
    pct = 100.0 * (a["n_expected"] - a["n_missing_slots"]) / a["n_expected"]
    GREEN, RED, GREY = "#1D9E75", "#D64541", "#666666"

    with PdfPages(out_path) as pdf:
        # ---- Page 1: summary + heatmap + monthly bars -----------------
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Image download completeness report",
                     fontsize=16, fontweight="bold", x=0.06, ha="left", y=0.96)
        fig.text(0.06, 0.915,
                 "%s   |   %s to %s   |   %d-min cadence, daily window %s-%s (%s)"
                 % (a["source"], a["first"], a["last"], a["cadence"],
                    _fmt(a["window"][0]), _fmt(a["window"][1]), a["tz"]),
                 fontsize=8.5, color=GREY)

        tiles = [("Complete", "%.1f%%" % pct, GREEN if pct >= 95 else RED),
                 ("Images on disk", str(a["n_parsed"]), "#222222"),
                 ("Missing", str(a["n_missing_slots"]), RED),
                 ("Gap events", str(len(a["gap_events"])), "#222222"),
                 ("Days fully absent", str(len(a["missing_days"])), "#222222")]
        for i, (label, val, color) in enumerate(tiles):
            x = 0.06 + i * 0.185
            fig.patches.append(FancyBboxPatch(
                (x, 0.80), 0.165, 0.085, transform=fig.transFigure,
                boxstyle="round,pad=0.008", fc="#f4f4f2", ec="none"))
            fig.text(x + 0.012, 0.862, label, fontsize=8, color=GREY)
            fig.text(x + 0.012, 0.815, val, fontsize=15,
                     fontweight="bold", color=color)

        # Heatmap
        ax1 = fig.add_axes([0.07, 0.42, 0.88, 0.30])
        import numpy as np
        grid = np.zeros((len(slots), len(rows)))
        for j, (_d, present, _m) in enumerate(rows):
            for i, s in enumerate(slots):
                grid[i, j] = 1 if s in present else 0
        from matplotlib.colors import ListedColormap
        ax1.pcolormesh(grid, cmap=ListedColormap([RED, "#c7e6da"]),
                       vmin=0, vmax=1)
        ax1.set_yticks([i + 0.5 for i in range(len(slots))])
        ax1.set_yticklabels([_fmt(s) for s in slots], fontsize=7)
        month_ticks = [j for j, (d, _p, _m) in enumerate(rows)
                       if d.day == 1 or j == 0]
        ax1.set_xticks(month_ticks)
        ax1.set_xticklabels([rows[j][0].strftime("%b %Y")
                             for j in month_ticks], fontsize=7)
        ax1.invert_yaxis()
        ax1.set_title("Availability heatmap (green = present, red = missing)",
                      fontsize=10, loc="left")
        for sp in ax1.spines.values():
            sp.set_visible(False)
        ax1.tick_params(length=0)

        # Monthly bars
        ax2 = fig.add_axes([0.07, 0.08, 0.88, 0.24])
        per_month = {}
        for d, _p, missing in rows:
            k = d.strftime("%Y-%m")
            per_month[k] = per_month.get(k, 0) + len(missing)
        keys = sorted(per_month)
        vals = [per_month[k] for k in keys]
        bars = ax2.bar(range(len(keys)), vals, color=RED, width=0.6)
        ax2.bar_label(bars, fontsize=8, color=GREY)
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels(
            [datetime.strptime(k, "%Y-%m").strftime("%b %Y") for k in keys],
            fontsize=8)
        ax2.set_title("Missing images per month", fontsize=10, loc="left")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.tick_params(axis="y", labelsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2+: gap-event table ---------------------------------
        ev = sorted(a["gap_events"], key=lambda g: -g[4])
        per_page = 38
        for p0 in range(0, len(ev), per_page):
            chunk = ev[p0:p0 + per_page]
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Gap events (largest first)%s"
                         % ("" if len(ev) <= per_page else
                            "  -  page %d of %d"
                            % (p0 // per_page + 1,
                               (len(ev) + per_page - 1) // per_page)),
                         fontsize=13, fontweight="bold",
                         x=0.06, ha="left", y=0.95)
            cells = [["%s %s" % (d0, _fmt(s0)), "%s %s" % (d1, _fmt(s1)),
                      str(n)] for d0, s0, d1, s1, n in chunk]
            ax = fig.add_axes([0.06, 0.04, 0.88, 0.86])
            ax.axis("off")
            tbl = ax.table(cellText=cells,
                           colLabels=["From", "Through", "Missing images"],
                           colWidths=[0.38, 0.38, 0.24],
                           cellLoc="left", loc="upper left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#e0e0e0")
                cell.set_height(0.022)
                if r == 0:
                    cell.set_facecolor("#f4f4f2")
                    cell.set_text_props(fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Image download completeness report"
        d["Creator"] = "GRIME AI"
    return out_path


def generate_gap_report(download_folder, tz: Optional[str] = None,
                        cadence_minutes: Optional[int] = None,
                        out_dir=None) -> Tuple[str, str]:
    """Analyze a download folder and write the HTML + CSV report into it."""
    a = analyze(download_folder, resolve_usgs_tz(tz), cadence_minutes)
    dest = out_dir or download_folder
    h, c = write_report(a, dest)
    try:
        write_pdf_report(a, dest)          # optional; needs matplotlib
    except Exception:
        pass                               # PDF is a bonus, never a failure
    return h, c


if __name__ == "__main__":
    args = sys.argv[1:]
    tz = None
    if "--tz" in args:
        i = args.index("--tz")
        tz = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit("usage: GRIME_AI_gap_report.py <folder|listing.txt> [--tz ZONE]")
    h, c = generate_gap_report(args[0], tz=tz,
                               out_dir=os.getcwd()
                               if not Path(args[0]).is_dir() else None)
    print("wrote", h, "and", c)
