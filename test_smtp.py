import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SENDER = os.getenv("SMTP_SENDER")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

def send_test_email():
    try:
        # Create the email
        msg = MIMEText("✅ This is a test email from Python SMTP script.")
        msg["Subject"] = "SMTP Test"
        msg["From"] = SMTP_SENDER
        msg["To"] = ADMIN_EMAIL

        # Connect to SMTP server
        print(f"Connecting to {SMTP_SERVER}:{SMTP_PORT}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_SENDER, ADMIN_EMAIL, msg.as_string())

        print(f"✅ Test email sent successfully to {ADMIN_EMAIL}")

    except Exception as e:
        print("❌ Email failed:", e)

if __name__ == "__main__":
    send_test_email()
 