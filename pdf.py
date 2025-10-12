import logging
import math
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
from src.chatbot import chatbot_bp
from src.chatbot_routes import chatbot_bp
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env file
load_dotenv()

if os.getenv("SENTRY_KEY") and os.getenv("SENTRY_PROJECT") and os.getenv("SENTRY_HOST"):
    sentry_sdk.init(
        dsn=f"https://{os.getenv('SENTRY_KEY')}@{os.getenv('SENTRY_HOST')}/{os.getenv('SENTRY_PROJECT')}",
        integrations=[FlaskIntegration()],
    )

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'b0609b9b73170e0d13c4ce616560cc8a316c14e93e4f54ab'
# Enable template auto-reload to refresh templates during development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=15)
# Disable caching for static files during development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')  # use your .env variable
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


 
# Elastic APM configuration
app.config["ELASTIC_APM"] = {
    "SERVICE_NAME": "global-uk-tariff",
    "SECRET_TOKEN": os.getenv("APM_TOKEN"),
    "SERVER_URL": "https://apm.elk.uktrade.digital",
    "ENVIRONMENT": os.getenv("ENV_NAME"),
}

# Disable caching for static files by setting autorefresh to True
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/", autorefresh=True)
apm = ElasticAPM(app)

# Generate a key for encryption (make sure to store this key securely for later decryption)
def generate_encryption_key():
    return Fernet.generate_key()

# Save the encryption key securely (You should store this somewhere safe and private)
def save_encryption_key(key):
    with open("encryption_key.key", "wb") as key_file:
        key_file.write(key)

# Load the encryption key (use this when you need to decrypt)
def load_encryption_key():
    with open("encryption_key.key", "rb") as key_file:
        return key_file.read()

# Encrypt the parts of the API key
def encrypt_api_key_parts(part1, part2, encryption_key):
    cipher = Fernet(encryption_key)
    encrypted_part1 = cipher.encrypt(part1.encode())
    encrypted_part2 = cipher.encrypt(part2.encode())
    
    # Save the encrypted parts to a file
    with open("encrypted_api_key_parts.txt", "wb") as encrypted_file:
        encrypted_file.write(encrypted_part1 + b'\n' + encrypted_part2)

# Decrypt the API key parts and combine them
def decrypt_api_key_parts(encrypted_file_path, encryption_key):
    cipher = Fernet(encryption_key)
    
    # Read the encrypted parts from the file
    with open(encrypted_file_path, "rb") as encrypted_file:
        encrypted_data = encrypted_file.read().split(b'\n')
        encrypted_part1 = encrypted_data[0]
        encrypted_part2 = encrypted_data[1]
    
    # Decrypt the parts
    decrypted_part1 = cipher.decrypt(encrypted_part1).decode()
    decrypted_part2 = cipher.decrypt(encrypted_part2).decode()
    
    # Combine the decrypted parts to form the API key
    return decrypted_part1 + decrypted_part2

#Encrypt and save API key parts if not already done (run this separately first)
#Example for encrypting and saving the API key parts:
part1 = "sk-proj-Jcgx4YzLqRFQq9F9MXwTfq2sk_q1HwfklOFIviaF45ockbK_K7HIUDUngcXZH6ka3jI0kxXYU1T3"
part2 = "BlbkFJjKGwgEWutYmHyXYAzNCozAUvZVgvDT8wzlhe2sNmugk628iD4XHcmflri96tUZbi-E4J24l14A"
encryption_key = generate_encryption_key()
save_encryption_key(encryption_key)
encrypt_api_key_parts(part1, part2, encryption_key)

# Load the encryption key
encryption_key = load_encryption_key()

# Decrypt and combine the API key parts
openai.api_key = decrypt_api_key_parts("encrypted_api_key_parts.txt", encryption_key)

thread_pool = ThreadPoolExecutor()

# Generate cache-busting version string
VERSION = f"?v={int(os.path.getmtime(__file__))}"  # Based on the file modification time

GLOBAL_TARIFF_FILE = 'global-uk-tariff.xlsx'

EXCHANGE_RATE_API_URL = "https://api.exchangerate-api.com/v4/latest/"
BASE_CURRENCY = "GBP"

# Example credentials for validation
users = {
    "tradesphere@admin": "password",
    "tradesphere@user2": "user2",   
    "tradesphere@user3": "user3",
    "tradesphere@user4": "user4",
    "tradesphere@user5": "user5",
    "tradesphere@user6": "user6",
    "tradesphere@user7": "user7",
    "tradesphere@user81": "user8",
    "tradesphere@user91": "user9",
    "tradesphere@user101": "user10",
    "tradesphere@user111": "user11",
    "tradesphere@user121": "user12",
    "tradesphere@user131": "user13"
}

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set session lifetime to 5 minutes
app.permanent_session_lifetime = timedelta(minutes=5)

@app.before_request
def before_request():
    # If the user is logged in, reset the session's expiration time
    if 'user' in session:
        session.permanent = True  # Make the session permanent so it respects the lifetime setting
    else:
        session.permanent = False
import psycopg2
# Get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

from flask import session, redirect, render_template, request
from preferential_folder_setup import create_user_folders_if_needed
#from db import get_db_connection  # make sure your DB connector is imported

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Connect to DB and check credentials
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT * FROM users_main 
                WHERE (email = %s OR phone = %s) AND password = %s
            """, (username, username, password))

            user = cur.fetchone()

            if user:
                session["user"] = username  # Set session

                # ✅ Call folder creation function here
                create_user_folders_if_needed(username)

                return redirect("/tariff")  # or wherever your dashboard starts
            else:
                return render_template("login.html", error="Invalid username or password!")

        finally:
            cur.close()
            conn.close()

    return render_template("login.html")


@app.route("/")
def home():
    # Redirect to login page if not logged in
    if "user" not in session:
        return redirect("/login")
    return redirect("/tariff")

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
        return redirect("/login")

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
    return redirect("/login")

@app.route("/chatbot.html")
def chatbot():
    return flask.render_template("chatbot.html", version=VERSION)

 

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

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_id = data.get("user_id", str(uuid.uuid4()))
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Detect HS intent (very basic check)
        hs_keywords = ["hs code", "tariff code", "commodity code"]
        is_hs_query = any(keyword in user_message.lower() for keyword in hs_keywords)

        # Session state
        session = user_sessions.get(user_id, {"hs_flow": False, "step": 0, "answers": []})

        if is_hs_query:
            # Reset flow if user triggers HS query again
            session["hs_flow"] = True
            session["step"] = 1
            session["answers"] = []
        elif session.get("hs_flow", False):
            # Continue flow
            session["step"] += 1
        else:
            # General question, no flow state needed
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_message}],
                max_tokens=300,
                temperature=0.6
            )
            reply = response['choices'][0]['message']['content']
            return jsonify({"response": reply, "user_id": user_id})

        # HS flow questions
        hs_questions = [
            "1. What is the product name?",
            "2. What material is it made of?",
            "3. What is its primary use or application?",
            "4. Are there any other names it’s known by? (Yes/No)",
        ]

        if session["step"] <= len(hs_questions):
            if session["step"] > 1:
                session["answers"].append(user_message)

            next_q = hs_questions[session["step"] - 1]
            user_sessions[user_id] = session
            return jsonify({"response": next_q, "user_id": user_id})
        else:
            # Final step: get HS code suggestion
            session["answers"].append(user_message)
            prompt = (
                "Determine the most accurate HS Code based on the following:\n"
                f"1. Product Name: {session['answers'][0]}\n"
                f"2. Material: {session['answers'][1]}\n"
                f"3. Usage: {session['answers'][2]}\n"
                f"4. Alternate Names: {session['answers'][3]}\n"
                "Respond in this format:\n"
                "🔍 Got it. Searching database...\n"
                "✅ The closest match is: **[HS CODE] – [Description]**"
            )

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.6,
            )

            reply = response['choices'][0]['message']['content']
            user_sessions[user_id] = {"hs_flow": False, "step": 0, "answers": []}
            return jsonify({"response": reply, "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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
    required_columns = ['item_number', 'description', 'value', 'currency', 'country_of_origin', 'commodity_code', 'weight']
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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        session['registration_data'] = {
            'company': request.form['company_name'],
            'email': request.form['email'],
            'phone': request.form.get('phone'),
            'plan': request.form['plan'],
            'raw_password': request.form['password']
        }

        # Redirect to PayPal
        if session['registration_data']['plan'] == 'basic':
            return redirect("https://www.paypal.com/ncp/payment/LV48XZ9GJA6J8")
        elif session['registration_data']['plan'] == 'pro':
            return redirect("https://www.paypal.com/ncp/payment/P9CSBWRRV9G3L")

    return render_template("register.html")
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
part1 = "sk-proj-Jcgx4YzLqRFQq9F9MXwTfq2sk_q1HwfklOFIviaF45ockbK_K7HIUDUngcXZH6ka3jI0kxXYU1T3"
part2 = "BlbkFJjKGwgEWutYmHyXYAzNCozAUvZVgvDT8wzlhe2sNmugk628iD4XHcmflri96tUZbi-E4J24l14A"
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

@app.route('/ask', methods=['POST'])
def ask():
    """Handles chatbot interaction with conversation context"""
    global cache

    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "Empty input received."})

    session_id = session.get('session_id', 'default')  # Assign a session ID (default for now)

    if session_id not in cache:
        cache[session_id] = []  # Create conversation history for this session

    conversation = cache[session_id]  # Retrieve conversation history

    # Append user's input to the conversation history
    conversation.append({"role": "user", "content": user_input})

    logger.info(f"Processing conversation history for session: {session_id}")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,  # Send full conversation history
            max_tokens=150
        )

        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"].strip()
        else:
            return jsonify({"error": "Unexpected OpenAI API response format."})

    except Exception as e:
        logger.error(f"Failed to get response from OpenAI: {e}")
        return jsonify({"error": "Failed to get response from OpenAI."})

    # Append AI response to conversation history
    conversation.append({"role": "assistant", "content": answer})

    # Save updated conversation in cache
    cache[session_id] = conversation
    save_cache()

    return jsonify({"response": answer})

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


app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
@chatbot_bp.route('/')
def chatbot_ui():
    return render_template('chatbot.html') 
if __name__ == "__main__":
    app.run(debug=True)