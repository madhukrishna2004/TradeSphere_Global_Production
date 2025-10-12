import logging
import math
import os
import re
import json
import pickle
import secrets
import string
import tempfile
import zipfile
import traceback
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from concurrent.futures.thread import ThreadPoolExecutor
from cryptography.fernet import Fernet

import psycopg2
import psycopg2.extras
import pandas as pd
import requests
import openai
import smtplib
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
import matplotlib.pyplot as plt
from flask import (
    Flask, request, Response, jsonify, render_template, send_from_directory, 
    redirect, session, flash, send_file, url_for
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from email.mime.text import MIMEText
from fpdf import FPDF
from whitenoise import WhiteNoise
import logging
import math
from hs_models import HSModelBundle
import os
from concurrent.futures.thread import ThreadPoolExecutor
import re
from elasticapm.contrib.flask import ElasticAPM
import flask
from flask import request, Response, jsonify, render_template, send_from_directory
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from whitenoise import WhiteNoise
import openai
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from src import decorators, utils
from flask import Flask, request, redirect, session, render_template, flash
import openai
import os
import pandas as pd
import requests
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from datetime import timedelta
from datetime import datetime
logger = logging.getLogger(__name__)
import json
import os
import openai
import pickle
import logging
import pyttsx3
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
#from src.chatbot import chatbot_bp
from src.chatbot_routes import chatbot_bp
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
import random
# Import your custom modules
# from hs_models import HSModelBundle
from src import decorators, utils
# from src.chatbot_routes import chatbot_bp
# from preferential_folder_setup import create_user_folders_if_needed
# from scheduler import process_completed_shift
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'fallback-secret-key-change-in-production'),
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=15),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true',
    SESSION_COOKIE_SAMESITE='Lax',
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=0,
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file upload
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'),
    PREFERRED_URL_SCHEME='https'  # Force HTTPS URLs
)

# Initialize extensions
db = SQLAlchemy(app)

# Configure WhiteNoise for static files
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/", autorefresh=True)

# Initialize thread pool
thread_pool = ThreadPoolExecutor(max_workers=4)

# Generate cache-busting version string
VERSION = f"?v={int(os.path.getmtime(__file__))}"

# Constants
GLOBAL_TARIFF_FILE = 'global-uk-tariff.xlsx'
EXCHANGE_RATE_API_URL = "https://api.exchangerate-api.com/v4/latest/"
BASE_CURRENCY = "GBP"
ALLOWED_EXTENSIONS = {'wav', 'aiff', 'aifc', 'flac', 'xlsx', 'xls'}
CACHE_FILE = 'cache.pkl'

# SMTP configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.zoho.in")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "noreply@trade-sphereglobal.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Initialize cache
cache = {}

# Use SQLAlchemy for connection pooling instead of psycopg2.pool
# SQLAlchemy handles connection pooling internally

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# If you must use the encryption approach, use environment variables for the parts:
if not openai.api_key:
    try:
        # Encryption functions for API keys
        def generate_encryption_key():
            return Fernet.generate_key()

        def save_encryption_key(key):
            with open("encryption_key.key", "wb") as key_file:
                key_file.write(key)

        def load_encryption_key():
            with open("encryption_key.key", "rb") as key_file:
                return key_file.read()

        def encrypt_api_key_parts(part1, part2, encryption_key):
            cipher = Fernet(encryption_key)
            encrypted_part1 = cipher.encrypt(part1.encode())
            encrypted_part2 = cipher.encrypt(part2.encode())
            
            with open("encrypted_api_key_parts.txt", "wb") as encrypted_file:
                encrypted_file.write(encrypted_part1 + b'\n' + encrypted_part2)

        def decrypt_api_key_parts(encrypted_file_path, encryption_key):
            cipher = Fernet(encryption_key)
            
            with open(encrypted_file_path, "rb") as encrypted_file:
                encrypted_data = encrypted_file.read().split(b'\n')
                encrypted_part1 = encrypted_data[0]
                encrypted_part2 = encrypted_data[1]
            
            decrypted_part1 = cipher.decrypt(encrypted_part1).decode()
            decrypted_part2 = cipher.decrypt(encrypted_part2).decode()
            
            return decrypted_part1 + decrypted_part2

        part1 = os.getenv('OPENAI_PART1')
        part2 = os.getenv('OPENAI_PART2')
        if part1 and part2:
            encryption_key = load_encryption_key()
            openai.api_key = decrypt_api_key_parts("encrypted_api_key_parts.txt", encryption_key)
    except Exception as e:
        logger.error(f"Failed to decrypt OpenAI API key: {e}")

# Database Models
class User(db.Model):
    __tablename__ = 'users_main'
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    plan = db.Column(db.String(50), nullable=False)
    payment_status = db.Column(db.String(50), default='pending')
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)
    otp_code = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class PasswordReset(db.Model):
    __tablename__ = 'password_resets'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users_main.id'), nullable=False)
    token = db.Column(db.String(100), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContactSubmission(db.Model):
    __tablename__ = 'contact_submissions'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    company = db.Column(db.String(120))
    message = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_cache():
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            logger.info("Cache loaded successfully")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Cache was corrupted. Resetting. Error: {e}")
            cache = {}
            save_cache()

def save_cache():
    global cache
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
            f.flush()
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Initialize cache
load_cache()

def get_exchange_rates(base_currency):
    response = requests.get(f"{EXCHANGE_RATE_API_URL}{base_currency}", timeout=10)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError("Failed to fetch exchange rates")

def process_excel(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    required_columns = ['item_number', 'description', 'value', 'currency', 
                       'country_of_origin', 'commodity_code', 'weight']
    
    if not all(col in df.columns for col in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required column(s): {', '.join(missing_columns)}")
    
    rates = get_exchange_rates(BASE_CURRENCY)['rates']
    
    df['value_gbp'] = df.apply(
        lambda row: row['value'] / rates.get(row['currency'], 1) if row['currency'] in rates else None,
        axis=1
    )
    
    df = df.dropna(subset=['value_gbp'])
    total_value = df['value_gbp'].sum()
    
    contributions = (
        df.groupby('country_of_origin')['value_gbp']
        .sum()
        .apply(lambda x: (x / total_value) * 100)
        .to_dict()
    )
    
    total_weight = df['weight'].sum()
    df['weight_contribution_percentage'] = (df['weight'] / total_weight) * 100
    
    return df, total_value, contributions, rates

def add_watermark(pdf):
    pdf.set_text_color(220, 220, 220)
    pdf.set_font("Arial", 'B', 50)
    pdf.set_xy(30, 130)
    pdf.cell(0, 20, "Tradesphere Global", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)

def generate_beautiful_pdf(data, total, contributions, rates, excel_file='processed_data.xlsx', 
                          filename_prefix='Enhanced_Summary_Report', agreement_type='EU-UK'):
    output_dir = 'src/pdf_report'
    os.makedirs(output_dir, exist_ok=True)
    pdf_output_file = os.path.join(output_dir, f"{filename_prefix}.pdf")
    
    required_columns = ['item_number', 'description', 'value_gbp', 'country_of_origin', 
                       'commodity_code', 'currency', 'weight', 'weight_contribution_percentage']
    
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Missing column: {column}")
    
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    add_watermark(pdf)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    try:
        excel_data = pd.read_excel(excel_file, header=None)
        headers = excel_data.iloc[0]
        values = excel_data.iloc[1]
        
        final_product = values[headers[headers == 'final_product'].index[0]]
        commodity = values[headers[headers == 'commodity'].index[0]]
        origin = values[headers[headers == 'origin'].index[0]]
    except (KeyError, IndexError):
        final_product = 'Final Product'
        commodity = 'Commodity'
        origin = 'Origin'
    
    # Title and header
    pdf.set_xy(10, 10)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="Summary Report", ln=True, align='C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 12)
    col_widths = [20, 50, 25, 40, 40, 20]
    headers = ["Item", "Description", "Value (GBP)", "Country of Origin", "Commodity", "Weight"]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 12, txt=header, border=1, align='C', fill=True)
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(240, 240, 240)
    fill = False
    
    for _, row in data.iterrows():
        fill = not fill
        pdf.cell(col_widths[0], 10, txt=str(row['item_number']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[1], 10, txt=row['description'], border=1, align='L', fill=fill)
        pdf.cell(col_widths[2], 10, txt=f"{row['value_gbp']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_widths[3], 10, txt=row['country_of_origin'], border=1, align='C', fill=fill)
        pdf.cell(col_widths[4], 10, txt=str(row['commodity_code']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[5], 10, txt=f"{row['weight']:.2f} kg", border=1, align='R', fill=fill)
        pdf.ln()
    
    # Add Summary Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Final Product Details:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Final Product = {final_product}", ln=True) 
    pdf.cell(0, 8, txt=f"Commodity = {commodity}", ln=True)
    pdf.cell(0, 8, txt=f"Principle of Origin = {origin}", ln=True)
    pdf.ln(10)
    
    # Rule of Origin Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Specific Rule of Origin", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, txt=f"{commodity} = {origin}")
    pdf.ln(10)
    
    # Conditional Text based on agreement type
    if agreement_type == 'EU-UK':
        eu_countries = [
            "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
            "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
            "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
        ]
        uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]
        partner_countries = eu_countries + uk_countries
        agreement_name = "EU-UK Preference trade agreement"
    else:  # Japan
        japan_countries = ["Japan"]
        uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]
        partner_countries = japan_countries + uk_countries
        agreement_name = "UK-Japan Preference trade agreement"
    
    # Calculate contributions
    partner_percentage = sum(
        percent for country, percent in contributions.items() 
        if country in partner_countries
    )
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in partner_countries
    )
    
    # Add logic for determining eligibility based on origin rules
    # (This would be similar to your existing logic but generalized)
    
    # Add more content based on your existing logic...
    
    pdf.output(pdf_output_file)
    return pdf_output_file

def send_email(to_email, subject, body_text):
    sender = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.zoho.in").strip()
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    
    if not sender or not password:
        raise ValueError("SMTP environment variables are missing")
    
    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {{ font-family: Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 0; color: #1f2937; }}
        .container {{ max-width: 600px; margin: 40px auto; background-color: #ffffff; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background-color: #0070f3; color: #ffffff; text-align: center; padding: 24px; font-size: 24px; font-weight: bold; }}
        .content {{ padding: 24px; line-height: 1.6; }}
        .button {{ display: inline-block; background-color: #0070f3; color: #ffffff; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; margin-top: 16px; }}
        .footer {{ text-align: center; font-size: 12px; color: #6b7280; padding: 16px; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">TradeSphere Global</div>
        <div class="content">{body_text}</div>
        <div class="footer">&copy; 2025 TradeSphere Global. All rights reserved.</div>
      </div>
    </body>
    </html>
    """
    
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = f"TradeSphere Global <{sender}>"
    msg["To"] = to_email
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        logger.info(f"Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise

def send_reset_email(to_email, reset_link):
    subject = "Password Reset - TradeSphere Global"
    body = f"""
    <p>Hello,</p>
    <p>You requested a password reset. Please click the button below to reset your password. This link is valid for 30 minutes.</p>
    <p style="text-align:center;"><a href="{reset_link}" class="button">Reset Password</a></p>
    <p>If you did not request this change, you can safely ignore this email.</p>
    <p>Regards,<br>TradeSphere Global Support Team</p>
    """
    
    try:
        send_email(to_email, subject, body)
        return True
    except Exception as e:
        logger.error(f"Failed to send reset email: {e}")
        return False

def get_commodity_details(commodity_code):
    try:
        df = pd.read_excel(GLOBAL_TARIFF_FILE)
        if 'commodity' not in df.columns:
            return None
        
        df['commodity'] = df['commodity'].astype(str).str.strip()
        df['description'] = df['description'].astype(str).str.strip()
        df['Product-specific rule of origin'] = df['Product-specific rule of origin'].astype(str).str.strip()
        
        matched = df[df['commodity'].str.startswith(str(commodity_code).strip())]
        
        if not matched.empty:
            return matched.to_dict(orient='records')
        
        return None
    except Exception as e:
        logger.error(f"Error getting commodity details: {e}")
        return None

# Middleware and Request Handling
@app.before_request
def before_request():
    # Redirect www to non-www
    if request.host.startswith('www.'):
        return redirect(f"https://{request.host[4:]}{request.path}", code=301)
    
    # Make session permanent and refresh on each request
    session.permanent = True
    session.modified = True
    
    # Check if user is logged in and reset session expiration
    if 'user' in session:
        # Verify user still exists in database
        try:
            user = User.query.filter_by(email=session['user']).first()
            if not user:
                session.pop('user', None)
                return redirect('/login')
        except Exception as e:
            logger.error(f"Error verifying user session: {e}")

@app.after_request
def after_request(response):
    # Add security headers
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Disable caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    
    return response

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

# Routes
@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("username")
        password = request.form.get("password")
        
        if not identifier or not password:
            return render_template("login.html", error="Username and password are required.")
        
        try:
            user = User.query.filter((User.email == identifier) | (User.phone == identifier)).first()
            
            if user and user.check_password(password):
                if not user.is_verified:
                    return render_template("login.html", error="Please verify your email before logging in.")
                
                session["user"] = user.email
                session["user_id"] = user.id
                
                logger.info(f"User {user.email} logged in successfully")
                return redirect("/dashboard")
            else:
                logger.warning(f"Failed login attempt for identifier: {identifier}")
                return render_template("login.html", error="Invalid credentials.")
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template("login.html", error="Something went wrong. Try again later.")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    user = session.get("user")
    session.clear()
    logger.info(f"User {user} logged out")
    return redirect("/")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        company = (request.form.get('company_name') or '').strip()
        email = (request.form.get('email') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        password = (request.form.get('password') or '').strip()
        confirm_password = (request.form.get('confirm_password') or '').strip()
        plan = (request.form.get('selected_plan') or '').strip()
        captcha_response = (request.form.get('g-recaptcha-response') or '').strip()
        
        if not all([company, email, phone, password, confirm_password, plan, captcha_response]):
            return render_template("register.html", error="All fields are required.")
        
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")
        
        # CAPTCHA validation
        secret_key = os.getenv('RECAPTCHA_SECRET_KEY', '6LdUYZkrAAAAAPoMmftPY91tslkdJ1wbsT1e6G_q')
        verify_url = 'https://www.google.com/recaptcha/api/siteverify'
        
        try:
            result = requests.post(
                verify_url,
                data={'secret': secret_key, 'response': captcha_response},
                timeout=5
            ).json()
        except requests.RequestException:
            return render_template("register.html", error="CAPTCHA service unavailable. Please try again later.")
        
        if not result.get('success'):
            return render_template("register.html", error="CAPTCHA verification failed. Please try again.")
        
        try:
            # Check if email or phone already exists
            existing_user = User.query.filter((User.email == email) | (User.phone == phone)).first()
            if existing_user:
                return render_template("register.html", error="Email or phone already registered.")
            
            # Hash password and generate OTP
            hashed_password = generate_password_hash(password)
            otp_code = str(random.randint(100000, 999999))
            otp_expiry = datetime.now() + timedelta(minutes=10)
            
            # Create new user
            new_user = User(
                company_name=company,
                email=email,
                phone=phone,
                password=hashed_password,
                plan=plan,
                otp_code=otp_code,
                otp_expiry=otp_expiry,
                is_verified=False
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            # Send OTP email
            try:
                send_email(
                    email,
                    "TradeSphere OTP Verification",
                    f"Hello {company},\n\nYour OTP code is: {otp_code}\n\nThis code is valid for 10 minutes."
                )
            except Exception as e:
                logger.error(f"Failed to send OTP email: {e}")
            
            # Set session and redirect to OTP verification
            session["pending_email"] = email
            return redirect(url_for("verify_otp"))
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            db.session.rollback()
            return render_template("register.html", error="Registration failed. Please try again.")
    
    return render_template("register.html")

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get("pending_email")
    if not email:
        return redirect("/register")
    
    if request.method == "POST":
        otp_input = (request.form.get("otp") or "").strip()
        
        try:
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return render_template("verify_otp.html", error="User not found")
            
            if otp_input == user.otp_code and datetime.now() < user.otp_expiry:
                user.is_verified = True
                user.otp_code = None
                user.otp_expiry = None
                db.session.commit()
                
                session.pop("pending_email", None)
                logger.info(f"User {email} verified successfully")
                return redirect("/login")
            else:
                return render_template("verify_otp.html", error="Invalid or expired OTP")
        except Exception as e:
            logger.error(f"OTP verification error: {e}")
            db.session.rollback()
            return render_template("verify_otp.html", error="Something went wrong. Please try again.")
    
    return render_template("verify_otp.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        try:
            if request.is_json:
                data = request.get_json()
                email = data.get("email")
            else:
                email = request.form.get("email")
            
            if not email:
                if request.is_json:
                    return jsonify({"error": "Email is required."}), 400
                return render_template("forgot_password.html", error="Email is required.")
            
            # Check if email exists
            user = User.query.filter_by(email=email).first()
            
            if not user:
                logger.warning(f"Password reset attempt for non-existent email: {email}")
                if request.is_json:
                    return jsonify({"error": "Email not registered."}), 400
                return render_template("forgot_password.html", error="Email not registered.")
            
            # Generate token + expiry
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=30)
            
            # Store reset token
            reset_entry = PasswordReset(
                user_id=user.id,
                token=token,
                expires_at=expires_at
            )
            
            db.session.add(reset_entry)
            db.session.commit()
            
            # Reset link
            reset_link = f"https://trade-sphereglobal.com/reset-password/{token}"
            
            # Send email
            if send_reset_email(email, reset_link):
                logger.info(f"Password reset email sent to {email}")
                if request.is_json:
                    return jsonify({"message": "A password reset link has been sent to your email."})
                return render_template("forgot_password.html", message="A password reset link has been sent to your email.")
            else:
                if request.is_json:
                    return jsonify({"error": "Could not send reset email."}), 500
                return render_template("forgot_password.html", error="Could not send reset email. Please try again later.")
                
        except Exception as e:
            logger.error(f"Forgot password error: {e}")
            db.session.rollback()
            if request.is_json:
                return jsonify({"error": "Something went wrong. Try again later."}), 500
            return render_template("forgot_password.html", error="Something went wrong. Try again later.")
    
    return render_template("forgot_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        reset_entry = PasswordReset.query.filter_by(token=token).first()
        
        if not reset_entry:
            return render_template("reset_password.html", error="Invalid or expired reset link.")
        
        if reset_entry.expires_at < datetime.utcnow():
            # Delete expired token
            db.session.delete(reset_entry)
            db.session.commit()
            return render_template("reset_password.html", error="Invalid or expired reset link.")
        
        if request.method == "POST":
            new_password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            
            if not new_password or new_password != confirm_password:
                return render_template("reset_password.html", token=token, error="Passwords do not match.")
            
            hashed_pw = generate_password_hash(new_password)
            
            # Update password
            user = User.query.get(reset_entry.user_id)
            user.password = hashed_pw
            db.session.commit()
            
            # Delete used token
            db.session.delete(reset_entry)
            db.session.commit()
            
            logger.info(f"Password reset successfully for user ID: {user.id}")
            return redirect(url_for("login"))
            
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        db.session.rollback()
        return render_template("reset_password.html", error="Something went wrong. Try again later.")
    
    return render_template("reset_password.html", token=token)

@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect("/login")
    
    return render_template("dashboard.html", user=session.get("user"))

# Add all your other routes here (tariff, api endpoints, etc.)
# Make sure to use proper error handling and database connection management

# ... [Include all your other routes with proper error handling]

from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash
from datetime import datetime

from flask import Flask, render_template, request, redirect, session, url_for
from datetime import datetime

from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash
from datetime import datetime
#`from your_db_module import get_db_connection  # make sure it's correctly imported
import requests
from flask import request, render_template, redirect, session
from werkzeug.security import generate_password_hash
from datetime import datetime
import random, smtplib
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import requests

# Get values
SMTP_SENDER = os.getenv("SMTP_SENDER")
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

print("Sender:", SMTP_SENDER)
print("Username:", SMTP_USERNAME)
print("Password:", SMTP_PASSWORD)

import os
import smtplib
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ✅ Email sending (Zoho SMTP Example)
import os
import smtplib
from email.mime.text import MIMEText

def send_email(to_email, subject, body_text):
    sender = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.zoho.in").strip()
    smtp_port = int(os.getenv("SMTP_PORT", 587))

    if not sender or not password:
        raise ValueError("SMTP environment variables are missing")

    # HTML Email Template (TradeSphere Global Theme)
    body = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{
          font-family: 'Inter', sans-serif;
          background-color: #f4f6f9;
          margin: 0;
          padding: 0;
          color: #1f2937;
        }}
        .container {{
          max-width: 600px;
          margin: 40px auto;
          background-color: #ffffff;
          border-radius: 16px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.1);
          overflow: hidden;
        }}
        .header {{
          background-color: #0070f3;
          color: #ffffff;
          text-align: center;
          padding: 24px;
          font-size: 24px;
          font-weight: bold;
        }}
        .content {{
          padding: 24px;
          line-height: 1.6;
        }}
        .button {{
          display: inline-block;
          background-color: #0070f3;
          color: #ffffff;
          text-decoration: none;
          padding: 12px 24px;
          border-radius: 8px;
          font-weight: 600;
          margin-top: 16px;
        }}
        .footer {{
          text-align: center;
          font-size: 12px;
          color: #6b7280;
          padding: 16px;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">TradeSphere Global</div>
        <div class="content">
          {body_text}
        </div>
        <div class="footer">
          &copy; 2025 TradeSphere Global. All rights reserved.
        </div>
      </div>
    </body>
    </html>
    """

    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] = f"TradeSphere Global <{sender}>"
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print("⚠️ Failed to send email:", e)
        raise

import psycopg2
# Get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        print("ℹ️ Received registration POST request")

        # 🔹 Extract and sanitize form data safely
        company = (request.form.get('company_name') or '').strip()
        email = (request.form.get('email') or '').strip()
        phone = (request.form.get('phone') or '').strip()
        password = (request.form.get('password') or '').strip()
        confirm_password = (request.form.get('confirm_password') or '').strip()
        plan = (request.form.get('selected_plan') or '').strip()
        captcha_response = (request.form.get('g-recaptcha-response') or '').strip()

        print(f"🔹 Form data: company='{company}', email='{email}', phone='{phone}', plan='{plan}'")

        # ✅ Basic field validation
        if not all([company, email, phone, password, confirm_password, plan, captcha_response]):
            print("⚠️ Validation failed: Missing fields")
            return render_template("register.html", error="All fields are required.")

        # 🔐 CAPTCHA validation
        secret_key = '6LdUYZkrAAAAAPoMmftPY91tslkdJ1wbsT1e6G_q'
        verify_url = 'https://www.google.com/recaptcha/api/siteverify'
        try:
            print("ℹ️ Verifying CAPTCHA")
            result = requests.post(
                verify_url,
                data={'secret': secret_key, 'response': captcha_response},
                timeout=5
            ).json()
        except requests.RequestException as ex:
            print("⚠️ CAPTCHA request failed:", ex)
            return render_template("register.html", error="CAPTCHA service unavailable. Please try again later.")

        if not result.get('success'):
            print("⚠️ CAPTCHA verification failed")
            return render_template("register.html", error="CAPTCHA verification failed. Please try again.")

        # ✅ Password match check
        if password != confirm_password:
            print("⚠️ Passwords do not match")
            return render_template("register.html", error="Passwords do not match.")

        # ✅ Database operations
        conn = None
        cur = None
        try:
            print("ℹ️ Connecting to database")
            conn = get_db_connection()
            cur = conn.cursor()

            # Check if email or phone already exists
            cur.execute("SELECT id FROM users_main WHERE email = %s OR phone = %s", (email, phone))
            if cur.fetchone():
                print("⚠️ Email or phone already registered")
                return render_template("register.html", error="Email or phone already registered.")

            # Hash password and generate OTP
            hashed_password = generate_password_hash(password)
            created_at = datetime.now()
            otp_code = str(random.randint(100000, 999999))
            otp_expiry = datetime.now() + timedelta(minutes=10)

            # Insert user into DB
            print("ℹ️ Inserting user into DB")
            cur.execute("""
                INSERT INTO users_main 
                (company_name, email, phone, password, plan, created_at, otp_code, otp_expiry, is_verified)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE)
            """, (company, email, phone, hashed_password, plan, created_at, otp_code, otp_expiry))
            conn.commit()
            print("✅ User inserted successfully")

            # Send OTP email
            try:
                print(f"ℹ️ Sending OTP email to {email}")
                send_email(
                    email,
                    "TradeSphere OTP Verification",
                    f"Hello {company},\n\nYour OTP code is: {otp_code}\n\nThis code is valid for 10 minutes."
                )
            except Exception:
                print("⚠️ OTP email sending failed but continuing registration")

            # Set session and redirect to OTP verification
            session["pending_email"] = email
            print(f"ℹ️ Redirecting to verify_otp for {email}")
            return redirect(url_for("verify_otp"))

        except Exception as e:
            print("🔴 DB INSERT ERROR:", e)
            return render_template("register.html", error="Registration failed: " + str(e))
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
            print("ℹ️ Database connection closed")

    # GET request
    print("ℹ️ Rendering registration page (GET)")
    return render_template("register.html")


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get("pending_email")
    if not email:
        return redirect("/register")

    if request.method == "POST":
        otp_input = (request.form.get("otp") or "").strip()

        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT otp_code, otp_expiry FROM users_main WHERE email=%s", (email,))
            user = cur.fetchone()

            if not user:
                return render_template("verify_otp.html", error="User not found")

            otp_code, otp_expiry = user

            if otp_input == otp_code and datetime.now() < otp_expiry:
                cur.execute("UPDATE users_main SET is_verified=TRUE, otp_code=NULL, otp_expiry=NULL WHERE email=%s", (email,))
                conn.commit()
                session.pop("pending_email", None)
                print(f"✅ OTP verified for {email}")
                return redirect("/login")
            else:
                print(f"⚠️ Invalid or expired OTP for {email}")
                return render_template("verify_otp.html", error="Invalid or expired OTP")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    return render_template("verify_otp.html")

@app.route("/accessibility", strict_slashes=False)
def accessibility():
    return flask.render_template("accessibility.html", version=VERSION)

@app.route("/tariff")
@decorators.cache_without_request_args(
    q=utils.DEFAULT_FILTER, p=utils.DEFAULT_PAGE, n=utils.DEFAULT_SAMPLE_SIZE
)

@decorators.compress_response
def tariff():
    # Ensure user is logged in
    if "user" not in session:
        return redirect("/")

    # Existing tariff logic
    data, total = utils.get_data_from_request()
    page = utils.get_positive_int_request_arg("p", utils.DEFAULT_PAGE)
    sample_size = utils.get_positive_int_request_arg("n", utils.DEFAULT_SAMPLE_SIZE)
    max_page = math.ceil(total / sample_size)

    return flask.render_template(
        "tariff.html",
        all_data=utils.get_data(get_all=True)[0],
        data=data,
        total=total,
        pages=utils.get_pages(page, max_page),
        page=page,
        max_page=total / sample_size,
        sample_size=sample_size,
        start_index=(sample_size * (page - 1)) + 1 if len(data) != 0 else 0,
        stop_index=sample_size * page if sample_size * page < total else total,
        version="1.0",  # Replace with the actual version variable if needed
    )



@app.route("/api/global-uk-tariff.csv")
@decorators.cache_without_request_args(
    q=utils.DEFAULT_FILTER, p=utils.DEFAULT_PAGE, n=utils.DEFAULT_SAMPLE_SIZE
)

@decorators.compress_response
def tariff_csv():
    filter_arg = request.args.get(utils.FILTER_ARG)
    data = utils.get_data_as_list(filter_arg)
    output = utils.format_data_as_csv(data)
    return flask.send_file(output, mimetype="text/csv")


@app.route("/api/global-uk-tariff.xlsx")
@decorators.cache_without_request_args(
    q=utils.DEFAULT_FILTER, p=utils.DEFAULT_PAGE, n=utils.DEFAULT_SAMPLE_SIZE
)
@decorators.compress_response
def tariff_xlsx():
    filter_arg = request.args.get(utils.FILTER_ARG)
    data = utils.get_data_as_list(filter_arg)
    output = utils.format_data_as_xlsx(data)
    return flask.send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/global-uk-tariff")
@decorators.cache_without_request_args(
    q=utils.DEFAULT_FILTER, p=utils.DEFAULT_PAGE, n=utils.DEFAULT_SAMPLE_SIZE
)
@decorators.compress_response
def tariff_api():
    data = utils.get_data_from_request(get_all=True)[0]
    return flask.jsonify(data)


@app.route("/tariff/metadata.json")
@decorators.cache_without_request_args()
@decorators.compress_response
def tariff_metadata():
    return flask.Response(
        flask.render_template("metadata.json"), mimetype="application/json",
    )


@app.route("/tariff/metadata.xml")
@decorators.cache_without_request_args()
@decorators.compress_response
def dcat_metadata():
    return flask.Response(
        flask.render_template("metadata.xml"), mimetype="application/rdf+xml",
    )

@app.route("/logout")
def logout():
    session.pop("user", None)  # Remove user from session
    return redirect("/")

 

 

@app.route('/pricings')
def pricings():
    return render_template('pricings.html')
 
@app.route('/origin')
def origin():
    return render_template('origin.html')

@app.route('/originjapan')
def origin_japan():
    return render_template('originjapan.html')

@app.route('/supplieraccess')
def supplieraccess():
    return render_template('supplieraccess.html')


user_sessions = {}
# HS Code guided questions
hs_questions = [
    "What is the product name?",
    "What material is it made of?",
    "What is its primary use or application?",
    "Are there any other names it’s known by? (Yes/No)"
]
import uuid
# Chat endpoint
user_sessions = {}  # Stores session context

 
@app.after_request
def set_cache_control_headers(response: Response):
    # Add headers to disable caching for all responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    # Prevent search engines from indexing the site
    response.headers["X-Robots-Tag"] = "index, nofollow"
    return response
    
# Redirect www to non-www
@app.before_request
def before_request():
    if request.host.startswith('www.'):
        # Redirect to non-www version with HTTPS
        return redirect(f"https://{request.host[4:]}{request.path}", code=301)

@app.after_request
def google_analytics(response: Response):
    try:
        kwargs = {}
        if request.accept_languages:
            try:
                kwargs["ul"] = request.accept_languages[0][0]
            except IndexError:
                pass

        if request.referrer:
            kwargs["dr"] = request.referrer

        path = request.path
        if request.query_string:
            path = path + f"?{request.query_string.decode()}"

        thread_pool.submit(
            utils.send_analytics,
            path=path,
            host=request.host,
            remote_addr=request.remote_addr,
            user_agent=request.headers["User-Agent"],
            **kwargs,
        )
    except Exception:  # We don't want to kill the response if GA fails.
        logger.exception("Google Analytics failed")
        pass
    return response
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory('static', 'sitemap.xml')
@app.route('/save-selected-data', methods=['POST'])
def save_selected_data():
    try:
        # Get data from the request
        data = request.json
        final_product = data.get('final_product')  # Final product name
        selected_data = data.get('selected_data', [])  # Selected data list

        # Basic validation: Check if the required fields are present
        if not final_product:
            return {"success": False, "error": "Final product name is required."}

        if not selected_data or not isinstance(selected_data, list):
            return {"success": False, "error": "No valid selected data provided."}

        # Prepare data for saving
        processed_data = [
            {
                "final_product": final_product,
                "commodity": item.get('hs_code', 'Unknown'),
                "origin": item.get('rule_of_origin', 'Unknown'),
                "description": item.get('description', 'Unknown')
            }
            for item in selected_data
        ]

        # File paths for saving
        json_file_path = os.path.join(os.getcwd(), 'processed_data.json')
        xlsx_file_path = os.path.join(os.getcwd(), 'processed_data.xlsx')

        # Save JSON data
        with open(json_file_path, 'w') as json_file:
            json.dump(processed_data, json_file, indent=4)

        # Save Excel data
        import pandas as pd
        df = pd.DataFrame(processed_data)
        df.to_excel(xlsx_file_path, index=False)

        # Respond with success and file paths
        return {
            "success": True,
            "message": "Data saved successfully.",
            "processed_data": processed_data,
            "json_file_path": json_file_path,
            "xlsx_file_path": xlsx_file_path
        }

    except Exception as e:
        # Catch and report any unexpected errors
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


# Function to fetch exchange rates
def get_exchange_rates(base_currency):
    response = requests.get(f"{EXCHANGE_RATE_API_URL}{base_currency}")
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError("Failed to fetch exchange rates.")

def process_excel(file):
    import pandas as pd

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check if all required columns are present
    required_columns = ['item_number', 'description', 'value', 'currency', 'country_of_origin', 'commodity_code', 'weight',]
    if not all(col in df.columns for col in required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        raise KeyError(f"Missing required column(s): {', '.join(missing_columns)}")

    # Fetch exchange rates
    rates = get_exchange_rates(BASE_CURRENCY)['rates']

    # Convert values to GBP
    df['value_gbp'] = df.apply(
        lambda row: row['value'] / rates.get(row['currency'], 1) if row['currency'] in rates else None,
        axis=1
    )

    # Drop rows with missing conversion
    df = df.dropna(subset=['value_gbp'])

    # Calculate total value in GBP
    total_value = df['value_gbp'].sum()

    # Calculate country contribution percentages
    contributions = (
        df.groupby('country_of_origin')['value_gbp']
        .sum()
        .apply(lambda x: (x / total_value) * 100)
        .to_dict()
    )

    # Add weight-based calculations
    total_weight = df['weight'].sum()
    df['weight_contribution_percentage'] = (df['weight'] / total_weight) * 100

    return df, total_value, contributions, rates
import os 
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

# Force Matplotlib to use a non-GUI backend
os.environ['MPLBACKEND'] = 'Agg'
def add_watermark(pdf):
    """Adds a watermark to the current page."""
    pdf.set_text_color(220, 220, 220)  # Light gray for watermark
    pdf.set_font("ARIAL", 'B', 50)
    pdf.set_xy(30, 130)  # Position watermark at the center
    pdf.cell(0, 20, "Tradesphere Global", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)  # Reset text color to black
def generate_beautiful_pdf(data, total, contributions, rates, excel_file='processed_data.xlsx', filename_prefix='Enhanced_Summary_Report'):
    print("Generating PDF report...")

    # Ensure the output directory exists
    output_dir = 'src/pdf_report'
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename for the PDF
    pdf_output_file = os.path.join(output_dir, f"{filename_prefix}.pdf")
    # Verify required columns
    required_columns = ['item_number', 'description', 'value_gbp', 'country_of_origin', 'commodity_code', 'currency', 'weight', 'weight_contribution_percentage']
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Missing column: {column}")

    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    add_watermark(pdf)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Reading dynamic headings from Excel
    try:
        excel_data = pd.read_excel(excel_file, header=None)  # Read without specifying the header
        headers = excel_data.iloc[0]  # First row as headers
        values = excel_data.iloc[1]  # Second row as values

        # Mapping dynamic headings to corresponding values
        final_product = values[headers[headers == 'final_product'].index[0]]
        commodity = values[headers[headers == 'commodity'].index[0]]
        origin = values[headers[headers == 'origin'].index[0]]

    except (KeyError, IndexError) as e:
        # Fallback values in case headings or values are missing
        final_product = 'Final Product'
        commodity = 'Commodity'
        origin = 'Origin'

    # Title and header
    pdf.set_xy(10, 10)
    pdf.set_font("ARIAL", 'B', 18)
    pdf.cell(0, 15, txt="Summary Report", ln=True, align='C')
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(200, 200, 200)  # Header color
    pdf.set_font("ARIAL", 'B', 12)
    col_widths = [20, 50, 25, 40, 40, 20, 30]  # Add space for the new column
    headers = ["Item", "Description", "Value (GBP)", "Country of Origin", "Commodity", "Weight"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 12, txt=header, border=1, align='C', fill=True)
    pdf.ln()

    # Table Rows
    pdf.set_font("ARIAL", size=10)
    pdf.set_fill_color(240, 240, 240)
    fill = False

    for _, row in data.iterrows():
        fill = not fill
        pdf.cell(col_widths[0], 10, txt=str(row['item_number']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[1], 10, txt=row['description'], border=1, align='L', fill=fill)
        pdf.cell(col_widths[2], 10, txt=f"{row['value_gbp']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_widths[3], 10, txt=row['country_of_origin'], border=1, align='C', fill=fill)
        pdf.cell(col_widths[4], 10, txt=str(row['commodity_code']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[5], 10, txt=f"{row['weight']:.2f} kg", border=1, align='R', fill=fill)
        #pdf.cell(col_widths[6], 10, txt=f"{row['weight_contribution_percentage']:.2f}%", border=1, align='R', fill=fill)
        pdf.ln()

    # Add Summary Section
    pdf.ln(10)
    pdf.set_font("ARIAL", 'B', 14)
    #pdf.cell(0, 10, txt="Assembled Place = UK", ln=True)
    pdf.cell(0, 10, txt="Final Product Details:", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 8, txt=f"Final Product = {final_product}", ln=True) 
    pdf.cell(0, 8, txt=f"Commodity = {commodity}", ln=True)
    pdf.cell(0, 8, txt=f"Principle of Origin = {origin}", ln=True)
    pdf.ln(10)

    # Rule of Origin Section
    pdf.set_font("ARIAL", 'B', 12)
    pdf.cell(0, 10, txt="Specific Rule of Origin", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.multi_cell(0, 8, txt=f"{commodity} = {origin}")
    pdf.ln(10)

    # Conditional Text
    if 'CTH' in origin:
        pdf.multi_cell(0, 8, txt=(
            "According to CTH: CTH means production from non-originating materials of any heading, "
            "except that of the product; this means that any non-originating material used in the "
            "production of the product must be classified under a heading (4-digit level of the Harmonised System) "
            "other than that of the product (i.e. a change in heading).Since the commodity codes in"
            "the bill of materials are not equal to the first four digits of the final product's commodity code," 
            "this product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
        ))
    elif 'CTSH' in origin:
        pdf.multi_cell(0, 8, txt=(
            "CTSH means production from non-originating materials of any subheading, except that of the product; "
            "this means that any non-originating material used in the production of the product must be classified under "
            "a subheading (6-digit level of the Harmonised System) other than that of the product (i.e. a change in subheading)."
            "the bill of materials are not equal to the first six digits of the final product's commodity code," 
            "this product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
        ))

    elif 'CC' in origin:
        pdf.multi_cell(0, 8, txt=(
            "CC means production from non-originating materials of any Chapter, except that of the"
            "product; this means that any non-originating material used in the production of the product"
            "alignmust be classified under a Chapter (2-digit level of the Harmonised System) other than that of"
            "the product (i.e. a change in Chapter);"

        )) 
       
    



    # MaxNOM Rule
    pdf.set_text_color(0, 0, 0)

# Check for CTH rule (excluding non-originating active cathode materials)
    if "cathode" in origin:
        pdf.ln(10)
        pdf.set_font("ARIAL", size=9)
        pdf.cell(0, 10, txt="*Please make sure there is no active cathode material in the Bill of Materials to qualify for CTH rule.", ln=True)

    # Check for MaxNOM rule
    if "MaxNOM" in origin:
        pdf.ln(10)
        pdf.set_font("ARIAL", size=12)
        pdf.cell(0, 10, txt="Alternatively, according to MaxNOM Rule:", ln=True)




    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 10, txt=f"Total Value: {total:.2f} GBP", ln=True)
    pdf.cell(0, 10, txt=f"(Calculated using today's exchange rates to convert\n"
                        "non-local currencies into GBP, the currency of the assembled country)", ln=True)
    pdf.ln(5)

    # Contribution Breakdown
    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Contribution Breakdown", ln=True)
    #pdf.ln(5)
    pdf.set_font("ARIAL", size=10)
    for country, percentage in contributions.items():
        pdf.cell(0, 8, txt=f"{country}: {percentage:.2f}%", ln=True)
    pdf.ln(5)  

    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Exchange Rates Used:", ln=True)
    pdf.set_font("ARIAL", size=10)
    relevant_currencies = data['currency'].unique()
    for currency in relevant_currencies:
        rate = rates.get(currency)
        if rate:
            pdf.cell(0, 8, txt=f"{rate:.2f} {currency} = 1 GBP", ln=True)

    

            
    eu_countries = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
    ]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]

    uk_eu_percentage = sum(percent for country, percent in contributions.items() if country in eu_countries or country in uk_countries)
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in eu_countries and country not in uk_countries
    )
    
    filtered_contributions = {
        country: value for country, value in contributions.items()
        if country not in eu_countries and country not in uk_countries
    }
    
    highest_contributed_country = max(filtered_contributions, key=filtered_contributions.get, default="Unknown")

    max_nom_percentage = rest_percentage  # Adjust as needed

    # Add vertical spacing
    pdf.ln(10)

    # Check `origin` conditions
    if "wholly obtained" in origin.lower():
        message = (
            f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
            "The product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
        )

    elif "MaxNOM" in origin:
        # Extract the dynamic threshold for MaxNOM
        match = re.search(r"MaxNOM (\d+)\s?%", origin)
        if match:
            threshold = int(match.group(1))
            if max_nom_percentage < threshold:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid MaxNOM condition specified."

    elif "except from non-originating materials of headings" in origin:
        # Extract the range from the origin string
        match = re.search(r"headings (\d+)\.(\d+) to (\d+)\.(\d+)", origin)
        if match:
            # Normalize start and end of the range (e.g., 72.08 -> 7208)
            start_range = int(match.group(1) + match.group(2))
            end_range = int(match.group(3) + match.group(4))

            # Check each heading in the commodity list
            all_eligible = True  # Assume eligible unless proven otherwise
            for heading in commodity:
                # Extract the leading part of the heading (e.g., 72096060 -> 7209)
                normalized_heading = int(heading[:4])  # Use only the first 4 digits
                if start_range <= normalized_heading <= end_range:
                    all_eligible = False  # Found an ineligible heading
                    break

            # Determine the message based on eligibility
            if all_eligible:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid heading condition specified."


    elif "the value of non-originating materials" in origin:
        # Extract percentage and headings from origin
        match = re.search(r"(\d+)%.*headings ([\d.]+)(?: and ([\d.]+))?", origin)
        if match:
            specified_percentage = int(match.group(1))  # Extract percentage threshold
            headings = [match.group(2).eplace('.', '')]  # Normalize heading1
            if match.group(3):
                headings.append(match.group(3).replace('.', ''))  # Normalize heading2 if present

            compliance = True
            for heading in headings:
                # Check if the heading exists in the commodity dictionary
                if heading in commodity and commodity[heading] <= specified_percentage:
                    continue
                compliance = False
                break

            if compliance:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid origin condition specified."

    else:
        # Generic fallback for unspecified conditions
        if max_nom_percentage < 50:
            message = (
                f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                "The product is eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
            )
        else:
            message = (
                f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                "The product is not eligible under the EU-UK Preference trade agreement for Zero or reduced Duty while importing."
            )



    pdf.set_font("ARIAL", 'I', 11)
    pdf.multi_cell(0, 8, txt=message)

    # Pie Chart for Contributions
    pdf.add_page()
    add_watermark(pdf) 
    pdf.set_font("ARIAL", 'B', 14)
    pdf.cell(0, 10, txt="Pie Chart of Contributions", ln=True)
    pdf.ln(5)

    eu_countries = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
    ]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK","United Kingdom","Great Britain"]

    # Combine EU and UK contributions
    uk_eu_percentage = sum(percent for country, percent in contributions.items() if country in eu_countries or country in uk_countries)
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in eu_countries and country not in uk_countries
    )
    labels = ['Other Countries', 'UK & EU']
    sizes = [rest_percentage, uk_eu_percentage]
    colors = ['#FF9999', '#66B2FF']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(edgecolor='black'))
    plt.axis('equal')
    plt.title('Contribution Breakdown', fontsize=14)
    pie_chart_path = 'pdf_report/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    pdf.image(pie_chart_path, x=10, y=60, w=180)

    # Add Final Note
    pdf.add_page()
    add_watermark(pdf) 
    pdf.ln(10)
    pdf.set_text_color(255, 0, 0)  # Red for notes
    pdf.set_font("ARIAL", 'B', 10)
    pdf.multi_cell(0, 8, txt=(
        "Note: Please note that this calculation assumes that all items within the EU/UK "
        "have valid preference origin statements from the suppliers."
    ))
       
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("ARIAL", 'B', 10)
    pdf.cell(0, 10, txt="Additionally, you can apply for a binding origin decision at HMRC:", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 10, txt="https://www.gov.uk/guidance/apply-for-a-binding-origin-information-decision", ln=True)

    pdf.set_font("ARIAL", 'B', 10)
    pdf.cell(0, 10, txt="Apply for an Advance Tariff Ruling:", ln=True)
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("ARIAL", size=10)

    # Adding a clickable link
    url = "https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling#apply-for-an-advance-tariff-ruling"
    pdf.cell(0, 10, txt="Go to Website", ln=True, link=url)
    
    pdf.set_text_color(0, 0, 0)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("ARIAL", 'I', 10)
    pdf.cell(0, 10, txt=f"Report generated on: {current_datetime}", ln=True)


    # Save PDF
    pdf.output(pdf_output_file)
    print(f"PDF report generated: {pdf_output_file}")
    return pdf_output_file

import zipfile
from werkzeug.utils import secure_filename
import tempfile

@app.route('/process-file', methods=['POST'])
def process_file():
    print("Received a file upload request.")

    if 'files' not in request.files:
        print("No file part.")
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        print("No selected files.")
        return jsonify({'error': 'No selected files'}), 400

    processed_files = []
    errors = []

    for file in files:
        if file:
            try:
                print(f"Processing file: {file.filename}")

                # Process the Excel file
                df, total_value, contributions, rates = process_excel(file)

                # Generate the PDF with the processed data
                pdf_output_file = generate_beautiful_pdf(
                    df, total_value, contributions, rates,
                    filename_prefix=f"Enhanced_Summary_Report_{secure_filename(file.filename)}"
                )

                # Ensure the file is not already in the list
                if pdf_output_file not in processed_files:
                    processed_files.append(pdf_output_file)
                    print(f"PDF generated: {pdf_output_file}")

            except KeyError as e:
                error_msg = f"File {file.filename}: Missing required columns: {str(e)}"
                print(f"KeyError: {error_msg}")
                errors.append(error_msg)
            except ValueError as e:
                error_msg = f"File {file.filename}: Value error: {str(e)}"
                print(f"ValueError: {error_msg}")
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"File {file.filename}: Unexpected error: {str(e)}"
                print(f"Unexpected error: {error_msg}")
                errors.append(error_msg)

    if errors:
        return jsonify({'error': 'Errors occurred during processing.', 'details': errors}), 400

    if not processed_files:
        return jsonify({'error': 'No files were processed successfully.'}), 400

    # Create a temporary zip file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        zip_path = temp_zip.name
        print(f"Creating temporary zip file at: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in processed_files:
                # Ensure unique filenames in the zip archive
                zip_filename = os.path.basename(pdf_file)
                zipf.write(pdf_file, zip_filename)

    # Return the zip file's URL for download
    zip_filename = os.path.basename(zip_path)
    return jsonify({
        'message': 'Files processed successfully.',
        'download_url': f'/download-report/{zip_filename}'
    }), 200
def generate_beautiful_pdf_japan(data, total, contributions, rates, excel_file='processed_data.xlsx', filename_prefix='Enhanced_Summary_Report'):
    print("Generating PDF report...")

    # Ensure the output directory exists
    output_dir = 'src/pdf_report'
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename for the PDF
    pdf_output_file = os.path.join(output_dir, f"{filename_prefix}.pdf")
    # Verify required columns
    required_columns = ['item_number', 'description', 'value_gbp', 'country_of_origin', 'commodity_code', 'currency', 'weight', 'weight_contribution_percentage']
    for column in required_columns:
        if column not in data.columns:
            raise KeyError(f"Missing column: {column}")

    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    add_watermark(pdf)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Reading dynamic headings from Excel
    try:
        excel_data = pd.read_excel(excel_file, header=None)  # Read without specifying the header
        headers = excel_data.iloc[0]  # First row as headers
        values = excel_data.iloc[1]  # Second row as values

        # Mapping dynamic headings to corresponding values
        final_product = values[headers[headers == 'final_product'].index[0]]
        commodity = values[headers[headers == 'commodity'].index[0]]
        origin = values[headers[headers == 'origin'].index[0]]

    except (KeyError, IndexError) as e:
        # Fallback values in case headings or values are missing
        final_product = 'Final Product'
        commodity = 'Commodity'
        origin = 'Origin'

    # Title and header
    pdf.set_xy(10, 10)
    pdf.set_font("ARIAL", 'B', 18)
    pdf.cell(0, 15, txt="Summary Report", ln=True, align='C')
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(200, 200, 200)  # Header color
    pdf.set_font("ARIAL", 'B', 12)
    col_widths = [20, 50, 25, 40, 40, 20, 30]  # Add space for the new column
    headers = ["Item", "Description", "Value (GBP)", "Country of Origin", "Commodity", "Weight"]

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 12, txt=header, border=1, align='C', fill=True)
    pdf.ln()

    # Table Rows
    pdf.set_font("ARIAL", size=10)
    pdf.set_fill_color(240, 240, 240)
    fill = False

    for _, row in data.iterrows():
        fill = not fill
        pdf.cell(col_widths[0], 10, txt=str(row['item_number']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[1], 10, txt=row['description'], border=1, align='L', fill=fill)
        pdf.cell(col_widths[2], 10, txt=f"{row['value_gbp']:.2f}", border=1, align='R', fill=fill)
        pdf.cell(col_widths[3], 10, txt=row['country_of_origin'], border=1, align='C', fill=fill)
        pdf.cell(col_widths[4], 10, txt=str(row['commodity_code']), border=1, align='C', fill=fill)
        pdf.cell(col_widths[5], 10, txt=f"{row['weight']:.2f} kg", border=1, align='R', fill=fill)
        #pdf.cell(col_widths[6], 10, txt=f"{row['weight_contribution_percentage']:.2f}%", border=1, align='R', fill=fill)
        pdf.ln()

    # Add Summary Section
    pdf.ln(10)
    pdf.set_font("ARIAL", 'B', 14)
    #pdf.cell(0, 10, txt="Assembled Place = UK", ln=True)
    pdf.cell(0, 10, txt="Final Product Details:", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 8, txt=f"Final Product = {final_product}", ln=True) 
    pdf.cell(0, 8, txt=f"Commodity = {commodity}", ln=True)
    pdf.cell(0, 8, txt=f"Principle of Origin = {origin}", ln=True)
    pdf.ln(10)

    # Rule of Origin Section
    pdf.set_font("ARIAL", 'B', 12)
    pdf.cell(0, 10, txt="Specific Rule of Origin", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.multi_cell(0, 8, txt=f"{commodity} = {origin}")
    pdf.ln(10)

    # Conditional Text
    if 'CTH' in origin:
        pdf.multi_cell(0, 8, txt=(
            "According to CTH: CTH means production from non-originating materials of any heading, "
            "except that of the product; this means that any non-originating material used in the "
            "production of the product must be classified under a heading (4-digit level of the Harmonised System) "
            "other than that of the product (i.e. a change in heading).Since the commodity codes in"
            "the bill of materials are not equal to the first four digits of the final product's commodity code," 
            "this product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
        ))
    elif 'CTSH' in origin:
        pdf.multi_cell(0, 8, txt=(
            "CTSH means production from non-originating materials of any subheading, except that of the product; "
            "this means that any non-originating material used in the production of the product must be classified under "
            "a subheading (6-digit level of the Harmonised System) other than that of the product (i.e. a change in subheading)."
            "the bill of materials are not equal to the first six digits of the final product's commodity code," 
            "this product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
        ))

    elif 'CC' in origin:
        pdf.multi_cell(0, 8, txt=(
            "CC means production from non-originating materials of any Chapter, except that of the"
            "product; this means that any non-originating material used in the production of the product"
            "alignmust be classified under a Chapter (2-digit level of the Harmonised System) other than that of"
            "the product (i.e. a change in Chapter);"

        )) 
       
    



    # MaxNOM Rule
    pdf.set_text_color(0, 0, 0)

# Check for CTH rule (excluding non-originating active cathode materials)
    if "cathode" in origin:
        pdf.ln(10)
        pdf.set_font("ARIAL", size=9)
        pdf.cell(0, 10, txt="*Please make sure there is no active cathode material in the Bill of Materials to qualify for CTH rule.", ln=True)

    # Check for MaxNOM rule
    if "MaxNOM" in origin:
        pdf.ln(10)
        pdf.set_font("ARIAL", size=12)
        pdf.cell(0, 10, txt="Alternatively, according to MaxNOM Rule:", ln=True)




    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 10, txt=f"Total Value: {total:.2f} GBP", ln=True)
    pdf.cell(0, 10, txt=f"(Calculated using today's exchange rates to convert\n"
                        "non-local currencies into GBP, the currency of the assembled country)", ln=True)
    pdf.ln(5)

    # Contribution Breakdown
    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Contribution Breakdown", ln=True)
    #pdf.ln(5)
    pdf.set_font("ARIAL", size=10)
    for country, percentage in contributions.items():
        pdf.cell(0, 8, txt=f"{country}: {percentage:.2f}%", ln=True)
    pdf.ln(5)  

    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Exchange Rates Used:", ln=True)
    pdf.set_font("ARIAL", size=10)
    relevant_currencies = data['currency'].unique()
    for currency in relevant_currencies:
        rate = rates.get(currency)
        if rate:
            pdf.cell(0, 8, txt=f"{rate:.2f} {currency} = 1 GBP", ln=True)

    

            
    japan_countries= [
        "Japan"
    ]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]

    uk_japan_percentage = sum(percent for country, percent in contributions.items() if country in japan_countries or country in uk_countries)
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in japan_countries and country not in uk_countries
    )
    
    filtered_contributions = {
        country: value for country, value in contributions.items()
        if country not in japan_countries and country not in uk_countries
    }
    
    highest_contributed_country = max(filtered_contributions, key=filtered_contributions.get, default="Unknown")

    max_nom_percentage = rest_percentage  # Adjust as needed

    # Add vertical spacing
    pdf.ln(10)

    # Check `origin` conditions
    if "wholly obtained" in origin.lower():
        message = (
            f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
            "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
        )

    elif "MaxNOM" in origin:
        # Extract the dynamic threshold for MaxNOM
        match = re.search(r"MaxNOM (\d+)\s?%", origin)
        if match:
            threshold = int(match.group(1))
            if max_nom_percentage < threshold:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid MaxNOM condition specified."

    elif "except from non-originating materials of headings" in origin:
        # Extract the range from the origin string
        match = re.search(r"headings (\d+)\.(\d+) to (\d+)\.(\d+)", origin)
        if match:
            # Normalize start and end of the range (e.g., 72.08 -> 7208)
            start_range = int(match.group(1) + match.group(2))
            end_range = int(match.group(3) + match.group(4))

            # Check each heading in the commodity list
            all_eligible = True  # Assume eligible unless proven otherwise
            for heading in commodity:
                # Extract the leading part of the heading (e.g., 72096060 -> 7209)
                normalized_heading = int(heading[:4])  # Use only the first 4 digits
                if start_range <= normalized_heading <= end_range:
                    all_eligible = False  # Found an ineligible heading
                    break

            # Determine the message based on eligibility
            if all_eligible:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid heading condition specified."

    elif "RVC" in origin:
    # Extract the required RVC percentage
        match = re.search(r"RVC (\d+)\s?%\s?\(FOB\)", origin)
        if match:
            threshold = int(match.group(1))

            # Calculate the originating content (UK + EU)
            originating_percentage = sum(
                percent for country, percent in contributions.items() if country in japan_countries or country in uk_countries
            )

            # Calculate the non-originating content
            non_originating_percentage = 100 - originating_percentage

            # Calculate RVC using the 'total' as the reference
            rvc_percentage = (total - (non_originating_percentage / 100 * total)) / total * 100

            if rvc_percentage >= threshold:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid RVC condition specified."


    elif "the value of non-originating materials" in origin:
        # Extract percentage and headings from origin
        match = re.search(r"(\d+)%.*headings ([\d.]+)(?: and ([\d.]+))?", origin)
        if match:
            specified_percentage = int(match.group(1))  # Extract percentage threshold
            headings = [match.group(2).replace('.', '')]  # Normalize heading1
            if match.group(3):
                headings.append(match.group(3).replace('.', ''))  # Normalize heading2 if present

            compliance = True
            for heading in headings:
                # Check if the heading exists in the commodity dictionary
                if heading in commodity and commodity[heading] <= specified_percentage:
                    continue
                compliance = False
                break

            if compliance:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
                )
        else:
            message = "Invalid origin condition specified."

    else:
        # Generic fallback for unspecified conditions
        if max_nom_percentage < 50:
            message = (
                f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                "The product is eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
            )
        else:
            message = (
                f"Based on the findings, according to product-specific rule of origin of the final product: {origin}.\n"
                "The product is not eligible under the UK-Japan Preference trade agreement for Zero or reduced Duty while importing."
            )



    pdf.set_font("ARIAL", 'I', 11)
    pdf.multi_cell(0, 8, txt=message)

    # Pie Chart for Contributions
    pdf.add_page()
    add_watermark(pdf) 
    pdf.set_font("ARIAL", 'B', 14)
    pdf.cell(0, 10, txt="Pie Chart of Contributions", ln=True)
    pdf.ln(5)

    japan_countries = [
        "Japan"
    ]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK","United Kingdom","Great Britain"]

    # Combine EU and UK contributions
    uk_japan_percentage = sum(percent for country, percent in contributions.items() if country in japan_countries or country in uk_countries)
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in japan_countries and country not in uk_countries
    )
    labels = ['Other Countries', 'UK & Japan']
    sizes = [rest_percentage, uk_japan_percentage]
    colors = ['#FF9999', '#66B2FF']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(edgecolor='black'))
    plt.axis('equal')
    plt.title('Contribution Breakdown', fontsize=14)
    pie_chart_path = 'pdf_report/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    pdf.image(pie_chart_path, x=10, y=60, w=180)

    # Add Final Note
    pdf.add_page()
    add_watermark(pdf) 
    pdf.ln(10)
    pdf.set_text_color(255, 0, 0)  # Red for notes
    pdf.set_font("ARIAL", 'B', 10)
    pdf.multi_cell(0, 8, txt=(
        "Note: Please note that this calculation assumes that all items within the EU/UK "
        "have valid preference origin statements from the suppliers."
    ))
       
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("ARIAL", 'B', 10)
    pdf.cell(0, 10, txt="Additionally, you can apply for a binding origin decision at HMRC:", ln=True)
    pdf.set_font("ARIAL", size=10)
    pdf.cell(0, 10, txt="https://www.gov.uk/guidance/apply-for-a-binding-origin-information-decision", ln=True)

    pdf.set_font("ARIAL", 'B', 10)
    pdf.cell(0, 10, txt="Apply for an Advance Tariff Ruling:", ln=True)
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("ARIAL", size=10)

    # Adding a clickable link
    url = "https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling#apply-for-an-advance-tariff-ruling"
    pdf.cell(0, 10, txt="Go to Website", ln=True, link=url)
    
    pdf.set_text_color(0, 0, 0)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("ARIAL", 'I', 10)
    pdf.cell(0, 10, txt=f"Report generated on: {current_datetime}", ln=True)


    # Save PDF
    pdf.output(pdf_output_file)
    print(f"PDF report generated: {pdf_output_file}")
    return pdf_output_file

import zipfile
from werkzeug.utils import secure_filename
import tempfile

@app.route('/process-file-japan', methods=['POST'])
def process_file_japan():
    print("Received a file upload request.")

    if 'files' not in request.files:
        print("No file part.")
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        print("No selected files.")
        return jsonify({'error': 'No selected files'}), 400

    processed_files = []
    errors = []

    for file in files:
        if file:
            try:
                print(f"Processing file: {file.filename}")

                # Process the Excel file
                df, total_value, contributions, rates = process_excel(file)

                # Generate the PDF with the processed data
                pdf_output_file = generate_beautiful_pdf_japan(
                    df, total_value, contributions, rates,
                    filename_prefix=f"Enhanced_Summary_Report_{secure_filename(file.filename)}"
                )

                # Ensure the file is not already in the list
                if pdf_output_file not in processed_files:
                    processed_files.append(pdf_output_file)
                    print(f"PDF generated: {pdf_output_file}")

            except KeyError as e:
                error_msg = f"File {file.filename}: Missing required columns: {str(e)}"
                print(f"KeyError: {error_msg}")
                errors.append(error_msg)
            except ValueError as e:
                error_msg = f"File {file.filename}: Value error: {str(e)}"
                print(f"ValueError: {error_msg}")
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"File {file.filename}: Unexpected error: {str(e)}"
                print(f"Unexpected error: {error_msg}")
                errors.append(error_msg)

    if errors:
        return jsonify({'error': 'Errors occurred during processing.', 'details': errors}), 400

    if not processed_files:
        return jsonify({'error': 'No files were processed successfully.'}), 400

    # Create a temporary zip file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        zip_path = temp_zip.name
        print(f"Creating temporary zip file at: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for pdf_file in processed_files:
                # Ensure unique filenames in the zip archive
                zip_filename = os.path.basename(pdf_file)
                zipf.write(pdf_file, zip_filename)

    # Return the zip file's URL for download
    zip_filename = os.path.basename(zip_path)
    return jsonify({
        'message': 'Files processed successfully.',
        'download_url': f'/download-report/{zip_filename}'
    }), 200
@app.route('/download-report/<filename>', methods=['GET'])
def download_report(filename):
    # Serve the generated zip file for download
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)

@app.route('/fetch-hs-code', methods=['POST'])
def fetch_hs_code():
    data = request.json
    product_name = data.get('product_name')
    if not product_name:
        return jsonify({'error': 'Product name is required'}), 400

    try:
        # Query OpenAI for the HS Code
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract only the numeric HS Code for the product description."},
                {"role": "user", "content": f"Product: {product_name}"}
            ]
        )
        
        # Extract the content from OpenAI response
        gpt_output = response['choices'][0]['message']['content']

        # Use regex to extract numeric HS Code (6-10 digits)
        match = re.search(r'\b\d{6,10}\b', gpt_output)
        if match:
            hs_code = match.group(0)  # Extract matched HS Code
            return jsonify({'hs_code': hs_code})  # Return only the code
        else:
            return jsonify({'error': 'Unable to retrieve a valid HS Code. Please click the Fetch HS Code button again to try retrieving the code, or you can enter it manually if the correct one is not fetched.'}), 400

    except Exception as e:
        # Handle OpenAI or server errors
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

def get_commodity_details(commodity_code):
    df = pd.read_excel(GLOBAL_TARIFF_FILE)
    if 'commodity' not in df.columns:
        return None

    df['commodity'] = df['commodity'].astype(str).str.strip()
    df['description'] = df['description'].astype(str).str.strip()
    df['Product-specific rule of origin'] = df['Product-specific rule of origin'].astype(str).str.strip()

    matched = df[df['commodity'].str.startswith(str(commodity_code).strip())]

    if not matched.empty:
        return matched.to_dict(orient='records')

    return None

@app.route('/hs-code-info/<string:hs_code>', methods=['GET'])
def get_hs_code_info(hs_code):
    try:
        # Retrieve commodity details based on hs_code
        matched_commodities = get_commodity_details(hs_code)

        if not matched_commodities:
            return jsonify({"error": "No matching commodities found for HS Code."})

        # Fetch origin and other details
        data = []
        for commodity in matched_commodities:
            data.append({
                "hs_code": commodity.get('commodity'),
                "description": commodity.get('description'),
                "rule_of_origin": commodity.get('Product-specific rule of origin'),
                #"country_of_origin": commodity.get('country_of_origin')  # Assuming country_of_origin is the principal origin
            })

        return jsonify({"matched_commodities": data})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/terms-of-service')
def terms():
    return render_template('terms_of_service.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy-policy')
def privacy():
    return render_template('privacy_policy.html')



from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from models import db
# DB Config
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



class User(db.Model):
    __tablename__ = 'users_main'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    plan = db.Column(db.String(50), nullable=False)
    payment_status = db.Column(db.String(50), default='pending')
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw_password):
        self.password = generate_password_hash(raw_password)

from flask import Flask, render_template, request, redirect, session
import os


from werkzeug.security import generate_password_hash
from models import db, User

@app.route('/payment-success')
def payment_success():
    data = session.get('registration_data')
    if not data:
        return redirect('/register')

    # Save user in DB
    hashed_pw = generate_password_hash(data['raw_password'])

    user = User(
        company_name=data['company'],
        email=data['email'],
        phone=data['phone'],
        plan=data['plan'],
        password=hashed_pw
    )

    db.session.add(user)
    db.session.commit()

    # Pass to /thanks and clear session
    session['user_email'] = data['email']
    session['user_password'] = data['raw_password']
    session.pop('registration_data', None)

    return redirect('/thanks')

@app.route('/thanks')
def thanks():
    email = session.get('user_email')
    password = session.get('user_password')

    if not email or not password:
        return redirect('/register')

    session.pop('user_email', None)
    session.pop('user_password', None)
    return render_template("thanks.html", email=email, password=password)

# Define Table
class ContactSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    company = db.Column(db.String(120))
    message = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)


from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
import os

# POST API Route
@app.route("/submit_contact", methods=["POST"])
def submit_contact():
    data = request.json
    try:
        entry = ContactSubmission(
            name=data.get("name"),
            email=data.get("email"),
            company=data.get("company"),
            message=data.get("message")
        )
        db.session.add(entry)
        db.session.commit()
        return jsonify({"success": True}), 200
    except Exception as e:
        print("DB Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

from flask import Flask, request, jsonify, render_template_string

 
from flask import Flask, render_template, request, redirect, session
from dotenv import load_dotenv
import os

# ✅ Import the db and models
from models import db, User
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ✅ Correct way to initialize db
db.init_app(app)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # or store in .env
from models import  User


# Admin Panel Route
@app.route('/admin/contacts')
def admin_contacts():
    if request.args.get("password") != "Madhu":
        return "Unauthorized", 401


    contacts = ContactSubmission.query.order_by(ContactSubmission.submitted_at.desc()).all()

    html = '''
    <html>
    <head>
        <title>Contact Submissions</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f7fc; padding: 40px; color: #333; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            th, td { padding: 12px 16px; border-bottom: 1px solid #ddd; }
            th { background-color: #0073e6; color: white; text-align: left; }
            tr:hover { background-color: #f1f1f1; }
            h1 { color: #0073e6; }
        </style>
    </head>
    <body>
        <h1>📊 Contact Submissions</h1>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Company</th>
                    <th>Message</th>
                    <th>Submitted At</th>
                </tr>
            </thead>
            <tbody>
            {% for c in contacts %}
                <tr>
                    <td>{{ c.id }}</td>
                    <td>{{ c.name }}</td>
                    <td>{{ c.email }}</td>
                    <td>{{ c.company }}</td>
                    <td>{{ c.message }}</td>
                    <td>{{ c.submitted_at.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    '''
    return render_template_string(html, contacts=contacts)


# OpenAI API Key
part1 = "sk-proj-J3KNdrW4oiFipUAixcYCXQmKkjBzVbdTQz_AQFHQGuM4EZRWTyfeZR41_wJGTVldfy7iZyeEhNT3Blb"
part2 = "kFJnpC3aI2B1eLcfLdQoHKPEq0ZYJyV2EiHu2yPQE_Woe5SVg8lZoxi39IW-8UgdxkrAxD9XctY4A" 
encryption_key = generate_encryption_key()
save_encryption_key(encryption_key)
encrypt_api_key_parts(part1, part2, encryption_key)

# Load the encryption key
encryption_key = load_encryption_key()

# Decrypt and combine the API key parts
openai.api_key = decrypt_api_key_parts("encrypted_api_key_parts.txt", encryption_key)

# Cache Memory Setup (Pickle)
CACHE_FILE = 'cache.pkl'
cache = {}

# Directory setup for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'aiff', 'aifc', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from flask_sqlalchemy import SQLAlchemy

 
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_cache():
    """Load cache from file if it exists"""
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            logger.info("Cache loaded successfully.")
        except (EOFError, pickle.UnpicklingError):
            cache = {}  # Reset cache if corrupted
            save_cache()
            logger.warning("Cache was corrupted. Resetting.")

def save_cache():
    """Save cache to file"""
    global cache
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
        f.flush()  # Ensure data is written
    logger.info("Cache saved successfully.")

@app.route('/index')
def index():
    return render_template('index.html')

 

from flask import Flask
from .routes.preferential_origin import preferential_origin_bp
from preferential_folder_setup import create_user_folders_if_needed
from dotenv import load_dotenv
import os
# Auto-create necessary folders
#create_user_folders_if_needed(session["username"])

# Register blueprint
app.register_blueprint(preferential_origin_bp, url_prefix="/preferential-origin")

from scheduler import process_completed_shift
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(func=process_completed_shift, trigger="cron", hour="6,12,18,23", minute="59")  # Just before shift ends
scheduler.start()

# Clean shutdown
import atexit
atexit.register(lambda: scheduler.shutdown())


import os
from flask import request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_login import current_user, login_required

UPLOAD_FOLDER = os.path.join(app.root_path, 'static/profile_pics')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from db import get_db

# Define the user-identifying column per table
TABLE_USER_COLUMN = {
    "email_logs": "user_id",                     # Only this uses user_id
    "supplier_declarations": "username",
    "supplier_received": "username",
    "bom_uploads": "username",
    "preferential_results": "username",
    "ai_lookups": "username",
    "user_logs": "username",                     # For login tracking
}

def get_count(cur, table, identifier, conn):
    try:
        user_col = TABLE_USER_COLUMN.get(table, "username")
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {user_col} = %s", (identifier,))
        return cur.fetchone()[0]
    except Exception as e:
        conn.rollback()
        print(f"[WARN] get_count failed for {table}: {e}")
        return "coming_soon"


def get_last_login(cur, identifier, conn):
    try:
        cur.execute("SELECT login_time, ip_address FROM user_logs WHERE username = %s ORDER BY login_time DESC LIMIT 1", (identifier,))
        row = cur.fetchone()
        if row:
            return f"{row[0].strftime('%Y-%m-%d @ %I:%M %p')} from {row[1]}"
    except Exception as e:
        conn.rollback()
        print(f"[WARN] get_last_login failed: {e}")
    return "Unknown"


@app.route("/profile")
def profile_dashboard():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]
    conn = get_db()
    cur = conn.cursor()

    # Fetch user info
    cur.execute("SELECT email, phone, language FROM users_main WHERE email = %s", (username,))
    user_info = cur.fetchone() or ("", "", "English")

    # Pull stats
    stats = {
        "emails_sent": get_count(cur, "email_logs", username, conn),
        "declarations_sent": get_count(cur, "supplier_declarations", username, conn),
        "declarations_received": get_count(cur, "supplier_received", username, conn),
        "boms_input": get_count(cur, "bom_uploads", username, conn),
        "preferential_checked": get_count(cur, "preferential_results", username, conn),
        "ai_lookups": get_count(cur, "ai_lookups", username, conn),
        "last_login": get_last_login(cur, username, conn),
    }

    cur.close()
    conn.close()

    return render_template("profile_dynamic.html", user_info=user_info, stats=stats, username=username)

@app.route("/billing")
def billing_page():
    if "user" not in session:
        return redirect("/login")
    
    username = session["user"]
    conn = get_db()
    cur = conn.cursor()

    # Subscription Details
    cur.execute("SELECT plan_name, start_date, end_date, status, next_billing_date, payment_method FROM subscriptions WHERE user_email = %s", (username,))
    subscription = cur.fetchone()

    # Invoices
    cur.execute("SELECT invoice_number, amount, issue_date, due_date, paid, pdf_url FROM invoices WHERE user_email = %s ORDER BY issue_date DESC", (username,))
    invoices = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("billing.html", subscription=subscription, invoices=invoices)

@app.route("/guru-purnima")
def guru_purnima():
    return render_template("guru_purnima.html")

from src.chatbot_routes import chatbot_bp
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')



# 404 handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('src/pdf_report', exist_ok=True)
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)