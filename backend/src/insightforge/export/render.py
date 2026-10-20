"""Report export: Markdown download and PDF (WeasyPrint).

The PDF is rendered from the same Markdown the user sees, with the confidence labels as
coloured badges and `[UNVERIFIED]` claims in amber. WeasyPrint is imported lazily: it needs
the pango system library (`brew install pango` on macOS, `apt install libpango-1.0-0` on
Debian); without it, PDF export reports itself unavailable instead of crashing the API.
On macOS, Homebrew's lib directory is not on the library search path, so it is added before the import.
"""

import html
import os
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-right { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; } }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4mm; } h2 { font-size: 14pt; margin: 7mm 0 2mm; border-bottom: 1px solid #ddd; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 8.5pt; font-weight: 700; color: #fff; }
.HIGH { background: #1f7a3d; } .MEDIUM { background: #b26a00; } .LOW { background: #b3261e; }
.unverified { background: #fff3cd; color: #8a5a00; font-weight: 700; padding: 0 3px; border-radius: 3px; }
.cite { color: #555; font-size: 8.5pt; } .meta { color: #666; font-size: 9pt; margin-bottom: 6mm; }
"""

_md = MarkdownIt("commonmark")


class ExportUnavailable(RuntimeError):
    pass


def report_html(markdown: str, title: str, meta: str = "") -> str:
    body = _md.render(markdown)
    body = re.sub(r"<strong>Confidence: (HIGH|MEDIUM|LOW)</strong>",
                  r'<span class="badge \1">Confidence: \1</span>', body)
    body = body.replace("<strong>[UNVERIFIED]</strong>", '<span class="unverified">[UNVERIFIED]</span>')
    body = re.sub(r"(\[Source: [^\]]+\])", r'<span class="cite">\1</span>', body)
    meta_html = f'<p class="meta">{html.escape(meta)}</p>' if meta else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{meta_html}{body}</body></html>")


def _add_homebrew_libs() -> None:
    """Let WeasyPrint find Homebrew's pango (find_library reads this variable at call time)."""
    if sys.platform != "darwin":
        return
    dirs = [d for d in ("/opt/homebrew/lib", "/usr/local/lib") if Path(d).is_dir()]
    current = [p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p]
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(current + [d for d in dirs if d not in current])


def report_pdf(markdown: str, title: str, meta: str = "") -> bytes:
    _add_homebrew_libs()
    try:
        from weasyprint import HTML
    except OSError as exc:
        raise ExportUnavailable("PDF export needs the pango library on the server "
                                "(macOS: brew install pango; Debian: apt install libpango-1.0-0).") from exc
    return HTML(string=report_html(markdown, title, meta)).write_pdf()
