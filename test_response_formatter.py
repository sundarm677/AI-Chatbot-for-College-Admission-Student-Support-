"""
test_response_formatter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for response_formatter.py — Smart Campus v3

Run:  python test_response_formatter.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from response_formatter import (
    detect_device,
    format_response,
    trim_response_for_mobile,
    get_response_headers,
)

PASS = "✅"
FAIL = "❌"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


# ═══════════════════════════════════════════════════════════
# 1. Device detection
# ═══════════════════════════════════════════════════════════
print("\n── Device Detection ─────────────────────────────────")

MOBILE_UAS = [
    "Mozilla/5.0 (Linux; Android 13; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; Redmi Note 11) AppleWebKit/537.36 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Xiaomi MI 9T) AppleWebKit/537.36 Mobile Safari/537.36",
]
for ua in MOBILE_UAS:
    d = detect_device(ua)
    check(f"mobile UA → '{d}'", d == "mobile", ua[:50])

TABLET_UAS = [
    "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
]
for ua in TABLET_UAS:
    d = detect_device(ua)
    check(f"tablet UA → '{d}'", d == "tablet", ua[:50])

DESKTOP_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "",
]
for ua in DESKTOP_UAS:
    d = detect_device(ua)
    check(f"desktop UA → '{d}'", d == "desktop", ua[:50])


# ═══════════════════════════════════════════════════════════
# 2. Format response — plain text
# ═══════════════════════════════════════════════════════════
print("\n── Plain-text Formatting ───────────────────────────")

plain_short = "Hello! How can I help you today?"
check("short plain unchanged on desktop",
      format_response(plain_short, "desktop") == plain_short)
check("short plain unchanged on mobile",
      format_response(plain_short, "mobile") == plain_short)

# Long plain text should be wrapped on mobile
long_plain = "This is a very long response that goes on and on and should be wrapped because it exceeds the mobile line length limit of around fifty-five characters per line."
mobile_result = format_response(long_plain, "mobile")
lines = mobile_result.split("\n")
check("long plain wrapped on mobile (all lines ≤ 60 chars)",
      all(len(l) <= 60 for l in lines),
      f"max line={max(len(l) for l in lines)}")

desktop_result = format_response(long_plain, "desktop")
check("long plain NOT wrapped on desktop",
      "\n" not in desktop_result and len(desktop_result) >= len(long_plain) - 2)


# ═══════════════════════════════════════════════════════════
# 3. Format response — HTML
# ═══════════════════════════════════════════════════════════
print("\n── HTML Formatting ─────────────────────────────────")

admission_html = (
    "📄 <b>Admission Process</b><br><br>"
    "1. Fill the online application at our website<br>"
    "2. Upload documents: 10th, 12th marksheets, ID proof<br>"
    "3. Pay application fee: ₹500<br>"
    "4. Entrance test / merit-based selection<br>"
    "5. Attend counselling and confirm seat<br><br>"
    "📅 Admissions open: June – August every year<br>"
    "📧<a href='mailto:admission@jec.ac.in'>admission@jec.ac.in</a><br>"
    "📧<a href='mailto:info@jec.ac.in'>info@jec.ac.in</a>"
    "📧<a href='mailto:principal@jec.ac.in'>principal@jec.ac.in</a><br>"
    "📞 044-26300982, 26341264, 26390041"
)

for device in ("mobile", "tablet", "desktop"):
    r = format_response(admission_html, device)
    check(f"HTML admission response not empty on {device}", bool(r))
    check(f"HTML admission keeps <b> tags on {device}", "<b>" in r)
    check(f"HTML admission keeps <br> tags on {device}", "<br>" in r)

# No triple+ consecutive <br> in output
for device in ("mobile", "tablet", "desktop"):
    r = format_response(admission_html, device)
    check(f"no triple-br on {device}", "<br><br><br>" not in r)

# Menu emoji items get broken into lines on mobile
menu_html = (
    "👋 Hello! Welcome.<br><br>"
    "1️⃣ Student Enquiry 2️⃣ Faculty Enquiry 3️⃣ Parent Enquiry 4️⃣ Visitor Enquiry"
)
mobile_menu = format_response(menu_html, "mobile")
check("menu emojis broken onto separate lines on mobile",
      mobile_menu.count("️⃣") >= 4)


# ═══════════════════════════════════════════════════════════
# 4. Trim for mobile
# ═══════════════════════════════════════════════════════════
print("\n── Mobile Truncation ───────────────────────────────")

short_resp = "Short response."
check("short response not truncated on mobile",
      trim_response_for_mobile(short_resp, "mobile") == short_resp)

long_resp = ("<br>word " * 400)  # ~3200 chars
trimmed = trim_response_for_mobile(long_resp, "mobile", max_chars=1800)
check("long response truncated on mobile",
      len(trimmed) < len(long_resp))
check("truncated response ends with continuation hint",
      "keyword" in trimmed or "details" in trimmed)

check("desktop response never truncated",
      trim_response_for_mobile(long_resp, "desktop") == long_resp)
check("tablet response never truncated",
      trim_response_for_mobile(long_resp, "tablet") == long_resp)


# ═══════════════════════════════════════════════════════════
# 5. Response headers
# ═══════════════════════════════════════════════════════════
print("\n── Response Headers ────────────────────────────────")

for device in ("mobile", "tablet", "desktop"):
    h = get_response_headers(device)
    check(f"Cache-Control=no-store on {device}",
          "no-store" in h["Cache-Control"])
    check(f"X-Device-Type={device} in headers",
          h["X-Device-Type"] == device)
    check(f"Vary=User-Agent on {device}",
          "User-Agent" in h["Vary"])


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "═" * 52)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total  = len(results)
print(f"  Total: {total}   Passed: {passed}   Failed: {failed}")
if failed:
    print("\nFailed tests:")
    for r in results:
        if r[0] == FAIL:
            print(f"  {r[1]}  {r[2]}")
print("═" * 52 + "\n")
sys.exit(0 if failed == 0 else 1)
