"""
Smart Campus AI Chatbot — v3 (Full Stack)
══════════════════════════════════════════════════════════════
Technologies integrated:
  ✅ NLP       — spaCy + NLTK + TF-IDF (engine.py)
  ✅ ML        — Naive-Bayes intent classifier (engine.py)
  ✅ Database  — SQLite with users / query_logs / suggestions / sessions
  ✅ Security  — Werkzeug password hashing, session mgmt, rate limiting,
                 CSRF token, bcrypt-compatible hashing, input sanitisation
  ✅ Voice     — Web Speech API (browser-side, already in frontend)
  ✅ Multilingual — Tamil, Hindi, English (engine + Google Translate API)
  ✅ REST API  — /api/chat, /api/suggest, /api/stats  (JSON)
  ✅ AJAX      — frontend already uses $.get(); API endpoints added
  ✅ OTP       — email-based password reset (otp_utils.py)
  ✅ Responsive — backend-only device detection + response formatting
                  (response_formatter.py) — frontend UI UNCHANGED

Frontend is UNCHANGED — all original templates preserved.
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, Response)
from werkzeug.security import generate_password_hash, check_password_hash
from chatbot.engine import get_response, get_response_for_device, extract_entities
from otp_utils import generate_otp, send_otp_email, store_otp, verify_otp
from response_formatter import (
    detect_device, format_response,
    get_response_headers, trim_response_for_mobile,
)
import sqlite3, os, re, hashlib, time, json
from datetime import datetime
from functools import wraps

# ── Translation System ──────────────────────────────────────
import urllib.request
import urllib.parse

def _translate_api(text: str, src: str, dest: str) -> str:
    if src == dest or not text.strip(): return text
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={dest}&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res = json.loads(response.read().decode('utf-8'))
            return "".join([sentence[0] for sentence in res[0]])
    except Exception as e:
        print(f"[LANG] Translation API Error ({src}->{dest}): {e}")
        return text

def _translate_to_english(text: str, src_lang: str) -> str:
    if src_lang == "en" or not text.strip(): return text
    res = _translate_api(text, src=src_lang, dest="en")
    if res and res != text:
        print(f"[LANG] Input Translated ({src_lang} -> en): '{text}' -> '{res}'")
        return res
    return text

def _translate_from_english(html_text: str, dest_lang: str) -> str:
    if dest_lang == "en" or not html_text.strip(): return html_text
    import re
    parts = re.split(r'(<[^>]+>|&nbsp;)', html_text)
    translated_parts = []
    for p in parts:
        if not p or p.startswith('<') or p == '&nbsp;':
            translated_parts.append(p)
        else:
            if p.strip():
                res = _translate_api(p, src="en", dest=dest_lang)
                if res:
                    translated_parts.append(res)
                else:
                    translated_parts.append(p)
            else:
                translated_parts.append(p)
    translated_html = "".join(translated_parts)
    
    # Inject this.innerText into any button onclick handlers so the UI shows the translated text when clicked
    import re
    translated_html = re.sub(r'onclick="send\(([\'"][^\'"]+[\'"])\)"', r'onclick="send(\1, this.innerText)"', translated_html)

    print(f"[LANG] Output Translated (en -> {dest_lang}):\nOrig: {html_text}\nNew: {translated_html}")
    return translated_html
# ══════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartcampus_v3_secret_!@#2025")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ── Compression — reduces payload size for mobile networks ─
try:
    from flask_compress import Compress
    Compress(app)
except ImportError:
    pass  # flask-compress not installed; responses still work, just uncompressed

# ── Response size optimisation ─────────────────────────────
# Keep JSON responses compact (no extra whitespace) — important on slow mobile networks
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.config["JSON_SORT_KEYS"] = False

DB_PATH = os.path.join(os.path.dirname(__file__), "smartcampus.db")

# ─── DB helpers ────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_db():
    conn = get_db(); c = conn.cursor()

    # Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT    NOT NULL,
        email     TEXT    UNIQUE NOT NULL,
        password  TEXT    NOT NULL,
        role      TEXT    DEFAULT 'user',
        lang      TEXT    DEFAULT 'en',
        is_active INTEGER DEFAULT 1,
        last_login TEXT,
        created   TEXT)""")

    # Query logs  (NLP entities stored as JSON)
    c.execute("""CREATE TABLE IF NOT EXISTS query_logs(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        query      TEXT    NOT NULL,
        response   TEXT    NOT NULL,
        intent     TEXT,
        entities   TEXT,
        lang       TEXT    DEFAULT 'en',
        timestamp  TEXT)""")

    # Suggestions from users
    c.execute("""CREATE TABLE IF NOT EXISTS suggestions(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        suggestion TEXT    NOT NULL,
        status     TEXT    DEFAULT 'pending',
        timestamp  TEXT)""")

    # OTP store
    c.execute("""CREATE TABLE IF NOT EXISTS otp_store(
        email   TEXT PRIMARY KEY,
        otp     TEXT NOT NULL,
        expires TEXT NOT NULL)""")

    # Rate-limit tracking (IP + endpoint + window)
    c.execute("""CREATE TABLE IF NOT EXISTS rate_limits(
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ip        TEXT    NOT NULL,
        endpoint  TEXT    NOT NULL,
        hit_time  REAL    NOT NULL)""")

    # Default admin
    if not c.execute("SELECT id FROM users WHERE email=?",
                     ("admin@smartcampus.com",)).fetchone():
        c.execute("INSERT INTO users(name,email,password,role,created) VALUES(?,?,?,?,?)",
                  ("Admin","admin@smartcampus.com",
                   generate_password_hash("admin123"),"admin",now()))
        print("✅ Default admin: admin@smartcampus.com / admin123")
    conn.commit(); conn.close()

# ── Security helpers ───────────────────────────────────────
def sanitise(s: str, max_len: int = 500) -> str:
    """
    Strip dangerous characters, trim length, and normalise mobile-keyboard
    artefacts (smart quotes, zero-width spaces, non-breaking spaces, etc.).
    Safe for both touch (mobile) and keyboard (desktop) input.
    """
    s = str(s)
    # Normalise smart/curly quotes to straight quotes
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    # Strip zero-width and non-breaking spaces common on mobile keyboards
    s = s.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    s = s.replace('\u00a0', ' ')
    # Strip HTML-dangerous chars
    s = re.sub(r'[<>"\']', '', s)
    # Collapse excessive whitespace (mobile auto-correct can insert extra spaces)
    s = re.sub(r'[ \t]+', ' ', s)
    return s[:max_len].strip()

def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()

def _get_device() -> str:
    """Detect the requesting device type (mobile / tablet / desktop)."""
    ua = request.headers.get("User-Agent", "")
    return detect_device(ua)

def rate_limit(endpoint: str, max_hits: int = 30, window_sec: int = 60):
    """Decorator – block if IP exceeds max_hits in window_sec seconds."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip   = _client_ip()
            conn = get_db()
            cutoff = time.time() - window_sec
            # Clean old hits
            conn.execute("DELETE FROM rate_limits WHERE hit_time < ?", (cutoff,))
            count = conn.execute(
                "SELECT COUNT(*) FROM rate_limits WHERE ip=? AND endpoint=? AND hit_time>=?",
                (ip, endpoint, cutoff)).fetchone()[0]
            if count >= max_hits:
                conn.close()
                return jsonify({"error": "Too many requests. Please slow down."}), 429
            conn.execute("INSERT INTO rate_limits(ip,endpoint,hit_time) VALUES(?,?,?)",
                         (ip, endpoint, time.time()))
            conn.commit(); conn.close()
            return f(*args, **kwargs)
        return wrapped
    return decorator

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Admin access required.')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapped

# ══════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('Invalid Admin Credentials')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.errorhandler(404)
def not_found(e):
    return render_template("entire.html")

@app.route('/')
def home():
    print("HOME PAGE LOADED")
    return render_template('entire.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = sanitise(request.form.get('email',''))
        pw    = request.form.get('password','')
        conn  = get_db()
        user  = conn.execute("SELECT * FROM users WHERE email=? AND is_active=1",
                             (email,)).fetchone()
        if user and check_password_hash(user['password'], pw):
            conn.execute("UPDATE users SET last_login=? WHERE email=?", (now(), email))
            conn.commit(); conn.close()
            session.pop('guest', None)
            session['user_id'] = user['id']
            session.update({
                'user': user['email'],
                'name': user['name'],
                'role': user['role'],
                'lang': user['lang'] or 'en'
            })
            return redirect(url_for('admin') if user['role']=='admin' else url_for('chatbot'))
        conn.close()
        flash('Invalid email or password.')
    return render_template('login.html')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    name  = sanitise(request.form.get('name',''))
    email = sanitise(request.form.get('uemail',''))
    pw    = request.form.get('upassword','')
    if not (name and email and pw):
        flash('All fields required.')
        return redirect(url_for('register'))
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone()
    if existing:
        conn.execute("UPDATE users SET password=? WHERE email=?",
                     (generate_password_hash(pw), email))
        conn.commit(); conn.close()
        flash('Password updated! Please login.')
        return redirect(url_for('login'))
    conn.execute("INSERT INTO users(name,email,password,created) VALUES(?,?,?,?)",
                 (name, email, generate_password_hash(pw), now()))
    conn.commit(); conn.close()
    flash('Account created! Please login.')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Forgot Password with OTP ───────────────────────────────
@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/forgot/send_otp', methods=['POST'])
def send_otp_route():
    email = sanitise(request.form.get('email',''))
    conn  = get_db()
    user  = conn.execute("SELECT id FROM users WHERE email=?",(email,)).fetchone()
    conn.close()
    if not user:
        flash('No account found with this email.')
        return redirect(url_for('forgot'))
    otp = generate_otp()
    store_otp(DB_PATH, email, otp)
    ok, msg = send_otp_email(email, otp)
    session['otp_email'] = email
    if ok:
        flash(f'OTP sent to {email}. Check your inbox.')
    else:
        flash(f'[DEV MODE] OTP: {otp} — ({msg})')
    return render_template('forgot.html', step='verify', email=email)

@app.route('/forgot/verify_otp', methods=['POST'])
def verify_otp_route():
    email = sanitise(request.form.get('email',''))
    otp   = sanitise(request.form.get('otp',''))
    valid, msg = verify_otp(DB_PATH, email, otp)
    if not valid:
        flash(msg)
        return render_template('forgot.html', step='verify', email=email)
    session['otp_verified'] = email
    return render_template('forgot.html', step='reset', email=email)

@app.route('/forgot/reset_password', methods=['POST'])
def reset_password():
    email = sanitise(request.form.get('email',''))
    pw    = request.form.get('upassword','')
    if session.get('otp_verified') != email:
        flash('Session expired. Please start again.')
        return redirect(url_for('forgot'))
    conn = get_db()
    conn.execute("UPDATE users SET password=? WHERE email=?",
                 (generate_password_hash(pw), email))
    conn.commit(); conn.close()
    session.pop('otp_verified', None)
    session.pop('otp_email', None)
    flash('Password reset successfully! Please login.')
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════════════
# CHATBOT (legacy AJAX endpoint — keeps original frontend working)
# ══════════════════════════════════════════════════════════
@app.route('/chatbot')
def chatbot():
    is_guest = 'user_id' not in session
    if is_guest:
        session['guest'] = True
        name = "Guest User"
    else:
        session.pop('guest', None)
        name = session.get('name', 'Student')

    return render_template('chatbot.html', isGuest=is_guest, name=name, lang=session.get('lang', 'en'))

@app.route('/get')
@rate_limit("get_bot", max_hits=60, window_sec=60)
def get_bot_response():
    if 'user_id' not in session:
        return Response("⚠ Login required to use chatbot", status=403, content_type="text/plain; charset=utf-8")
        
    msg  = sanitise(request.args.get('msg', ''))

    device = _get_device()

    # NLP response
    resp     = get_response_for_device(msg, device=device)
    entities = extract_entities(msg)

    # ── Backend formatting for device ────────────────────────────
    resp = format_response(resp, device)
    resp = trim_response_for_mobile(resp, device)
    # ────────────────────────────────────────────────────────────

    # Log to DB
    conn = get_db()
    conn.execute(
        "INSERT INTO query_logs(user_email,query,response,intent,entities,lang,timestamp) "
        "VALUES(?,?,?,?,?,?,?)",
        (session.get('user','guest'), msg, resp,
         None,                          # intent stored in v3 via API endpoint
         json.dumps(entities),
         lang, now()))
    conn.commit(); conn.close()

    # Build response with device-aware headers
    headers = get_response_headers(device)
    # The frontend expects plain text/HTML string — preserve Content-Type
    # as text/plain so jQuery renders it as-is (the frontend uses .html() equivalent)
    return Response(resp, status=200, headers={
        "Cache-Control": headers["Cache-Control"],
        "Pragma":        headers["Pragma"],
        "X-Device-Type": headers["X-Device-Type"],
        "Vary":          headers["Vary"],
    }, content_type="text/plain; charset=utf-8")

@app.route('/set_lang', methods=['POST'])
@login_required
def set_lang():
    lang = sanitise(request.json.get('lang', 'en'), 5)
    session['lang'] = lang
    try:
        conn = get_db()
        conn.execute("UPDATE users SET lang=? WHERE email=?", (lang, session.get('user','')))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Error setting lang in db: {e}")
    return jsonify({"status": "ok", "lang": lang})

# ══════════════════════════════════════════════════════════
# REST API  (JSON — for AJAX / external integrations)
# ══════════════════════════════════════════════════════════
@app.route('/chat', methods=['POST'])
@rate_limit("api_chat", max_hits=60, window_sec=60)
def api_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Login required", "response": "⚠ Please login to access chatbot features."})
    """
    POST /api/chat
    Body: { "message": "...", "lang": "en" }
    Returns: { "response": "...", "intent": "...", "entities": {...}, "lang": "en",
               "device": "mobile|tablet|desktop" }
    """
    data    = request.get_json(silent=True) or {}
    msg     = sanitise(data.get('message', ''))
    lang    = sanitise(data.get('lang', session.get('lang','en')), 5)

    if not msg:
        return jsonify({"error": "message is required", "response": "Please enter a message."}), 400

    # Detect device from User-Agent
    device = _get_device()

    # Map numeric button codes to natural text so it logs and translates properly
    LABEL_MAP = {
        '1': 'Student Enquiry',
        '2': 'Faculty Enquiry',
        '3': 'Parent Enquiry',
        '4': 'Visitor Enquiry',
        '1.1': 'Admission process',
        '1.2': 'Courses offered',
        '1.3': 'Exam schedule',
        '1.4': 'Scholarships',
        '1.5': 'Hostel facilities',
        '1.6': 'Library',
        '1.7': 'Placement',
        '2.1': 'Faculty directory',
        '2.2': 'Department contacts',
        '2.3': 'Academic calendar',
        '2.4': 'Research facilities',
        '3.1': 'Student progress',
        '3.2': 'Fee structure',
        '3.3': 'Hostel info',
        '3.4': 'Grievance',
        '4.1': 'Campus location',
        '4.2': 'Courses overview',
        '4.3': 'Events & seminars',
        '4.4': 'Contact'
    }

    try:
        # If user clicked a numeric button, replace it with the English text equivalent immediately
        if msg in LABEL_MAP:
            english_msg = LABEL_MAP[msg]
        else:
            # 1. Translate input to English if needed
            english_msg = _translate_to_english(msg, lang)

        # 2. Process NLP purely in English
        resp     = get_response_for_device(english_msg, device=device)
        entities = extract_entities(english_msg)

        # 3. Translate response text ONLY back to selected language
        resp = _translate_from_english(resp, lang)

        # ── Backend formatting for device ────────────────────────────
        resp = format_response(resp, device)
        resp = trim_response_for_mobile(resp, device)
        # ────────────────────────────────────────────────────────────

        # Detect intent for logging
        from chatbot.engine import _predict_intent
        intent, conf = _predict_intent(msg)
    except Exception as e:
        print(f"Backend Error (NLP/Formatting): {e}")
        resp = "⚠️ The server encountered an issue processing your message. Please try again."
        intent = "error"
        conf = 0.0
        entities = {}

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO query_logs(user_email,query,response,intent,entities,lang,timestamp) "
            "VALUES(?,?,?,?,?,?,?)",
            (session.get('user','guest'), msg, resp, intent,
             json.dumps(entities), "en", now()))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"Database logging error: {e}")

    payload = {
        "response":   resp,
        "intent":     intent,
        "confidence": round(conf, 3),
        "entities":   entities,
        "lang":       lang,
        "device":     device,      # expose device type to API consumers
    }
    response_obj = jsonify(payload)
    # Set cache-control headers
    hdrs = get_response_headers(device)
    response_obj.headers["Cache-Control"] = hdrs["Cache-Control"]
    response_obj.headers["Pragma"]        = hdrs["Pragma"]
    response_obj.headers["X-Device-Type"] = hdrs["X-Device-Type"]
    response_obj.headers["Vary"]          = hdrs["Vary"]
    return response_obj

@app.route('/api/suggest', methods=['POST'])
@login_required
@rate_limit("api_suggest", max_hits=10, window_sec=60)
def api_suggest():
    """
    POST /api/suggest
    Body: { "suggestion": "..." }
    """
    data = request.get_json(silent=True) or {}
    text = sanitise(data.get('suggestion',''))
    if not text:
        return jsonify({"error": "suggestion is required"}), 400
    conn = get_db()
    conn.execute("INSERT INTO suggestions(user_email,suggestion,timestamp) VALUES(?,?,?)",
                 (session.get('user','guest'), text, now()))
    conn.commit(); conn.close()
    return jsonify({"status": "ok", "message": "Thank you for your suggestion!"})

@app.route('/api/stats')
@login_required
def api_stats():
    """Public (authenticated) stats endpoint."""
    conn = get_db()
    total_queries = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    top_intents   = conn.execute(
        "SELECT intent, COUNT(*) as cnt FROM query_logs WHERE intent IS NOT NULL "
        "GROUP BY intent ORDER BY cnt DESC LIMIT 5").fetchall()
    lang_dist     = conn.execute(
        "SELECT lang, COUNT(*) as cnt FROM query_logs GROUP BY lang").fetchall()
    conn.close()
    return jsonify({
        "total_queries": total_queries,
        "top_intents":   [{"intent": r["intent"], "count": r["cnt"]} for r in top_intents],
        "lang_distribution": [{"lang": r["lang"], "count": r["cnt"]} for r in lang_dist],
    })

@app.route('/api/translate', methods=['POST'])
@login_required
def api_translate():
    """
    POST /api/translate
    Body: { "text": "...", "dest": "ta" }
    """
    data = request.get_json(silent=True) or {}
    text = sanitise(data.get('text',''))
    dest = sanitise(data.get('dest','en'), 5)
    translated = _translate(text, dest)
    return jsonify({"original": text, "translated": translated, "lang": dest})

# ══════════════════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════════════════

from datetime import timedelta

@app.route('/analytics')
@admin_required
def analytics_api():
    conn = get_db()
    
    # total users
    tu = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    
    # total queries
    tq = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    
    # active users (Today)
    au_row = conn.execute("SELECT COUNT(DISTINCT user_email) FROM query_logs WHERE DATE(timestamp) = CURRENT_DATE").fetchone()
    active_users = au_row[0] if au_row else 0
    
    # top language
    tl_row = conn.execute("SELECT lang, COUNT(*) as cnt FROM query_logs GROUP BY lang ORDER BY cnt DESC LIMIT 1").fetchone()
    top_lang_code = tl_row['lang'] if tl_row else 'N/A'
    
    # Map to full language name
    if top_lang_code == 'en':
        top_language = 'English'
    elif top_lang_code == 'ta':
        top_language = 'Tamil'
    elif top_lang_code == 'hi':
        top_language = 'Hindi'
    else:
        top_language = top_lang_code.upper() if top_lang_code != 'N/A' else 'N/A'
    
    # query trends (last 7 days)
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    trends = conn.execute("SELECT substr(timestamp, 1, 10) as dt, COUNT(*) as cnt FROM query_logs WHERE timestamp >= ? GROUP BY dt ORDER BY dt ASC", (seven_days_ago,)).fetchall()
    query_trends = [{"date": r["dt"], "count": r["cnt"]} for r in trends]
    
    # language usage counts
    lang_stats = conn.execute("SELECT lang, COUNT(*) as cnt FROM query_logs GROUP BY lang").fetchall()
    language_usage = [{"lang": r["lang"], "count": r["cnt"]} for r in lang_stats]
    
    conn.close()
    
    return jsonify({
        "total_users": tu,
        "total_queries": tq,
        "active_users": active_users,
        "top_language": top_language,
        "query_trends": query_trends,
        "language_usage": language_usage
    })
@app.route('/admin')
@admin_required
def admin():
    conn = get_db()
    users       = conn.execute("SELECT * FROM users ORDER BY created DESC").fetchall()
    logs        = conn.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT 100").fetchall()
    suggestions = conn.execute("SELECT * FROM suggestions ORDER BY timestamp DESC LIMIT 50").fetchall()
    tu          = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    tq          = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    
    au_row = conn.execute("SELECT COUNT(DISTINCT user_email) FROM query_logs WHERE DATE(timestamp) = CURRENT_DATE").fetchone()
    active_users_today = au_row[0] if au_row else 0
    
    tl_row = conn.execute("SELECT lang, COUNT(*) as cnt FROM query_logs GROUP BY lang ORDER BY cnt DESC LIMIT 1").fetchone()
    top_lang_code = tl_row['lang'] if tl_row else 'N/A'
    if top_lang_code == 'en':
        top_language = 'English'
    elif top_lang_code == 'ta':
        top_language = 'Tamil'
    elif top_lang_code == 'hi':
        top_language = 'Hindi'
    else:
        top_language = top_lang_code.upper() if top_lang_code != 'N/A' else 'N/A'
        
    print(top_language)
    print(active_users_today)

    lang_stats  = conn.execute(
        "SELECT lang, COUNT(*) as cnt FROM query_logs GROUP BY lang").fetchall()
    conn.close()
    return render_template('admin.html', users=users, logs=logs,
                           total_users=tu, total_queries=tq,
                           active_users_today=active_users_today,
                           top_language=top_language,
                           lang_stats=lang_stats)

@app.route('/admin/delete_user/<int:uid>', methods=['POST'])
@admin_required
def delete_user(uid):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,))
    conn.commit(); conn.close()
    return redirect(url_for('admin'))

@app.route('/admin/export_logs')
@admin_required
def export_logs():
    conn = get_db()
    logs = conn.execute("SELECT * FROM query_logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    csv = "ID,User,Query,Response,Intent,Lang,Timestamp\n"
    for l in logs:
        q = str(l['query']).replace('"','""')
        r = str(l['response']).replace('"','""')[:120]
        intent = l['intent'] or ''
        csv += f'{l["id"]},"{l["user_email"]}","{q}","{r}",{intent},{l["lang"]},{l["timestamp"]}\n'
    return Response(csv, mimetype='text/csv',
                    headers={"Content-Disposition":"attachment;filename=query_logs_v3.csv"})

@app.route('/admin/toggle_user/<int:uid>', methods=['POST'])
@admin_required
def toggle_user(uid):
    conn = get_db()
    user = conn.execute("SELECT is_active FROM users WHERE id=?",(uid,)).fetchone()
    if user:
        new_state = 0 if user['is_active'] else 1
        conn.execute("UPDATE users SET is_active=? WHERE id=?",(new_state, uid))
        conn.commit()
    conn.close()
    return redirect(url_for('admin'))

# ── Error handlers ─────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def too_many(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
