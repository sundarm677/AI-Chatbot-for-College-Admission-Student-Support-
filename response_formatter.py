"""
response_formatter.py — Smart Campus v3
════════════════════════════════════════
Backend-only response formatting for cross-device compatibility.

Responsibilities:
  • Detect device type from User-Agent (mobile / tablet / desktop)
  • Reformat chatbot response text for the detected device
  • Break long HTML responses into short readable chunks for small screens
  • Keep responses lightweight (strip unnecessary whitespace, normalise spacing)
  • Never modify CSS, templates, or frontend assets

All formatting is done in Python before the response is sent over the wire.
The frontend receives already-formatted HTML — no JS changes required.
"""

import re
from typing import Literal

# ── Device detection ─────────────────────────────────────────────────────────

MOBILE_UA_PATTERNS = re.compile(
    r"(android|iphone|ipod|opera mini|iemobile|wpdesktop"
    r"|mobile|blackberry|bb10|windows phone"
    r"|symbian|palm|nokia|samsung|lg;|htc|sony|xiaomi|oppo|vivo"
    r"|bolt|boost|cricket|docomo|fone|huawei|lenovo|lg-|lg/"
    r"|nexus.+mobile|kindle|silk|playstation.+portable"
    r"|googlebot-mobile)",
    re.IGNORECASE,
)

TABLET_UA_PATTERNS = re.compile(
    r"(ipad|tablet|kindle|silk|playbook|nexus.(?!mobile)|gt-p|sm-t"
    r"|lenovo.+tab|tab.+lenovo|mediapad|nook|sch-i800|shw-m180)",
    re.IGNORECASE,
)

DeviceType = Literal["mobile", "tablet", "desktop"]


def detect_device(user_agent: str) -> DeviceType:
    """Return 'mobile', 'tablet', or 'desktop' based on User-Agent string."""
    if not user_agent:
        return "desktop"
    ua = user_agent.lower()
    if TABLET_UA_PATTERNS.search(ua):
        return "tablet"
    if MOBILE_UA_PATTERNS.search(ua):
        return "mobile"
    return "desktop"


# ── Line-length helpers ───────────────────────────────────────────────────────

# Maximum characters per line before we consider wrapping plain-text content
MAX_LINE_LEN: dict[DeviceType, int] = {
    "mobile":  55,
    "tablet":  90,
    "desktop": 160,
}

# Maximum characters for a single <br>-separated segment on mobile
MOBILE_SEGMENT_LIMIT = 120


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to a single space; trim leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long_plain_segment(segment: str, max_len: int) -> list[str]:
    """
    Split a plain-text segment (no HTML tags) that is longer than max_len
    at natural word boundaries.
    """
    if len(segment) <= max_len:
        return [segment]

    words = segment.split()
    lines, current = [], []
    current_len = 0
    for word in words:
        if current and current_len + 1 + len(word) > max_len:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += (1 if current_len else 0) + len(word)
    if current:
        lines.append(" ".join(current))
    return lines


# ── HTML-aware response reformatter ──────────────────────────────────────────

def _contains_html(text: str) -> bool:
    return bool(re.search(r"<[a-zA-Z/]", text))


def _ensure_br_after_each_bullet(html: str) -> str:
    """
    Guarantee that every bullet-like line (•, -, numbers like 1. / 1.1) is
    followed by a <br> so that items stack vertically on narrow screens.
    """
    # Already has <br> after bullet — leave it
    html = re.sub(
        r"((?:•|-|\d+[\.\d]*)\s+[^<\n]+?)(?<!<br>)(\s*(?=•|-|\d+[\.\d]*\s|<b>|$))",
        lambda m: m.group(1) + "<br>" + m.group(2),
        html,
    )
    return html


def _insert_section_breaks(html: str, device: DeviceType) -> str:
    """
    On mobile, add a blank <br> between major sections identified by <b> tags
    that act as section headers, so sections breathe on a small screen.
    On tablet, add a single <br> break.
    On desktop, leave as-is.
    """
    if device == "desktop":
        return html
    gap = "<br><br>" if device == "mobile" else "<br>"
    # Wrap each <b>…</b> header with a preceding blank line if not already
    html = re.sub(r"(?<!>)(<b>[^<]+</b>)", gap + r"\1", html)
    return html


def _trim_html_whitespace(html: str) -> str:
    """Remove redundant whitespace inside HTML while preserving tag structure."""
    # Remove leading/trailing spaces around <br>
    html = re.sub(r"\s*<br>\s*", "<br>", html)
    # Collapse double+ <br> to at most two
    html = re.sub(r"(<br>){3,}", "<br><br>", html)
    # Strip whitespace before/after <b> and </b>
    html = re.sub(r"\s*(</?b>)\s*", r" \1", html)
    return html.strip()


def _wrap_long_plain_text(html: str, device: DeviceType) -> str:
    """
    Walk through HTML and, for plain-text nodes longer than the device limit,
    wrap them at word boundaries using <br>.
    This avoids the browser having to do text-wrap on a tiny viewport.
    """
    if device == "desktop":
        return html
    max_len = MAX_LINE_LEN[device]

    # Tokenise into [tag, text, tag, text, …]
    parts = re.split(r"(<[^>]+>)", html)
    result = []
    for part in parts:
        if part.startswith("<"):
            result.append(part)
        else:
            # Plain text node — split into segments by existing newlines / <br>
            segments = re.split(r"\n", part)
            formatted_segments = []
            for seg in segments:
                seg = _normalise_whitespace(seg)
                if seg:
                    sub = _split_long_plain_segment(seg, max_len)
                    formatted_segments.extend(sub)
                else:
                    formatted_segments.append(seg)
            result.append("<br>".join(formatted_segments))
    return "".join(result)


def _mobile_compact_menu(html: str) -> str:
    """
    On mobile, convert inline menu items (1️⃣ … 2️⃣ … 3️⃣ …) that sit on a
    single line into one item per line using <br> for readability.
    The pattern matches emoji+number combos like 1️⃣ or plain "1." followed by text.
    """
    # Numbered emoji items in a single run — split them onto new lines
    html = re.sub(
        r"([1-4]️⃣\s+\S[^1-4\n<]{0,60}?)(?=[1-4]️⃣)",
        r"\1<br>",
        html,
    )
    return html


def _plain_text_to_mobile_lines(text: str, device: DeviceType) -> str:
    """
    For pure plain-text responses (no HTML), insert newlines/breaks so that
    lines don't exceed the device max width in characters.
    """
    if device == "desktop":
        return text
    max_len = MAX_LINE_LEN[device]
    lines = text.splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if len(line) <= max_len:
            result.append(line)
        else:
            result.extend(_split_long_plain_segment(line, max_len))
    return "\n".join(result)


# ── Public API ────────────────────────────────────────────────────────────────

def format_response(response: str, device: DeviceType) -> str:
    """
    Format a chatbot response string for the target device.

    Parameters
    ----------
    response : str
        Raw HTML or plain-text response from the chatbot engine.
    device   : DeviceType
        'mobile', 'tablet', or 'desktop'.

    Returns
    -------
    str
        Formatted response, ready to send to the client.
    """
    if not response:
        return response

    if _contains_html(response):
        # ── HTML response pipeline ─────────────────────────
        r = _trim_html_whitespace(response)
        r = _ensure_br_after_each_bullet(r)
        if device in ("mobile", "tablet"):
            r = _mobile_compact_menu(r)
            r = _insert_section_breaks(r, device)
            r = _wrap_long_plain_text(r, device)
            r = _trim_html_whitespace(r)   # final clean-up pass
        return r
    else:
        # ── Plain-text response pipeline ───────────────────
        r = _normalise_whitespace(response)
        r = _plain_text_to_mobile_lines(r, device)
        return r


def get_response_headers(device: DeviceType) -> dict:
    """
    Return HTTP response headers optimised for the device type.

    These headers hint to the client and any intermediate CDN/proxy to:
      - Deliver compressed payloads (Accept-Encoding handled by Flask/Werkzeug)
      - Not cache personalised chat responses
      - Allow cross-origin requests from the same origin only
    """
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma":        "no-cache",
        "X-Device-Type": device,
        "Vary":          "User-Agent",
    }
    # On mobile, hint that the response is lightweight text (UTF-8)
    if device == "mobile":
        headers["Content-Type"] = "text/plain; charset=utf-8"
    return headers


def trim_response_for_mobile(response: str, device: DeviceType,
                              max_chars: int = 1800) -> str:
    """
    If the formatted response exceeds max_chars on mobile, truncate at the
    last clean <br> boundary and append a continuation hint.

    Desktop and tablet responses are never truncated.
    """
    if device != "mobile" or len(response) <= max_chars:
        return response

    truncated = response[:max_chars]
    # Snap back to the last <br>
    last_br = truncated.rfind("<br>")
    if last_br > max_chars // 2:
        truncated = truncated[:last_br]

    return truncated + "<br><br>📱 <i>Type a keyword for more details.</i>"
