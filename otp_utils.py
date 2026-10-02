"""
OTP Email Utility — uses Python stdlib smtplib (no flask-mail needed)
Configure SMTP settings in config.py before use.
"""

import smtplib
import random
import string
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import sqlite3

# ─── SMTP Config — edit these ────────────────────────────
SMTP_HOST   = "smtp.gmail.com"
SMTP_PORT   = 587
SMTP_EMAIL  = "your_email@gmail.com"       # ← your Gmail
SMTP_PASS   = "your_app_password"          # ← Gmail App Password (not normal password)
SENDER_NAME = "Smart Campus"
# ─────────────────────────────────────────────────────────

def generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP."""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(to_email: str, otp: str) -> tuple[bool, str]:
    """
    Send OTP via Gmail SMTP.
    Returns (success: bool, message: str)
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Smart Campus — Password Reset OTP"
        msg["From"]    = f"{SENDER_NAME} <{SMTP_EMAIL}>"
        msg["To"]      = to_email

        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;background:#f0f4ff;padding:30px">
        <div style="max-width:480px;margin:auto;background:white;border-radius:12px;
                    padding:30px;box-shadow:0 4px 20px rgba(0,0,0,0.1)">
          <h2 style="color:#0072ff;margin-bottom:8px">🎓 Smart Campus</h2>
          <p style="color:#555">Password Reset Request</p>
          <hr style="border:1px solid #eee;margin:20px 0">
          <p style="color:#333">Your One-Time Password (OTP) is:</p>
          <div style="background:#f0f8ff;border:2px dashed #0072ff;border-radius:10px;
                      padding:20px;text-align:center;margin:20px 0">
            <span style="font-size:36px;font-weight:bold;letter-spacing:10px;color:#0072ff">
              {otp}
            </span>
          </div>
          <p style="color:#888;font-size:13px">⏰ This OTP is valid for <b>10 minutes</b>.</p>
          <p style="color:#888;font-size:13px">If you did not request this, ignore this email.</p>
          <hr style="border:1px solid #eee;margin:20px 0">
          <p style="color:#aaa;font-size:12px">Smart Campus — AI Chatbot System</p>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASS)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())

        return True, "OTP sent successfully."

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check your email/app password in otp_utils.py."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"


def store_otp(db_path: str, email: str, otp: str) -> None:
    """Store OTP in DB with expiry."""
    conn = sqlite3.connect(db_path)
    expires = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT OR REPLACE INTO otp_store(email, otp, expires)
        VALUES (?, ?, ?)
    """, (email, otp, expires))
    conn.commit()
    conn.close()


def verify_otp(db_path: str, email: str, otp: str) -> tuple[bool, str]:
    """Verify OTP — returns (valid, message)."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT otp, expires FROM otp_store WHERE email=?", (email,)
    ).fetchone()
    conn.close()

    if not row:
        return False, "No OTP found. Please request a new one."

    stored_otp, expires_str = row
    expires = datetime.strptime(expires_str, "%Y-%m-%d %H:%M:%S")

    if datetime.now() > expires:
        return False, "OTP expired. Please request a new one."
    if otp.strip() != stored_otp:
        return False, "Incorrect OTP. Please try again."

    return True, "OTP verified."
