import logging
import math
import os
from concurrent.futures.thread import ThreadPoolExecutor
import re
from elasticapm.contrib.flask import ElasticAPM
import flask
from flask import request, Response, jsonify, render_template
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
    "tradesphere@user2": "user2"
}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Validate credentials
        if username in users and users[username] == password:
            session["user"] = username  # Store username in the session
            return redirect("/tariff")  # Redirect to /tariff after login
        else:
            return render_template("login.html", error="Invalid username or password!")

    # Render the login page for GET requests
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

#@app.route("/tariff")
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

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/pricings')
def pricings():
    return render_template('pricings.html')

@app.route('/origin')
def origin():
    return render_template('origin.html')

@app.route('/supplieraccess')
def supplieraccess():
    return render_template('supplieraccess.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get the user message from the request
        user_message = request.json.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # OpenAI API call for the chatbot response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=200,
            temperature=0.6,
        )
        
        # Get the response from OpenAI
        bot_reply = response['choices'][0]['message']['content']
        
        return jsonify({"response": bot_reply})
    except Exception as e:
        logger.error(f"Error occurred while processing the chatbot message: {str(e)}")
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

@app.route('/save-selected-data', methods=['POST'])
def save_selected_data():
    try:
        # Get data from the request
        data = request.json
        final_product = data.get('final_product')  # Extract final_product
        selected_data = data.get('selected_data', [])  # Extract selected_data list, default to empty list

        if not selected_data:
            return {"success": False, "error": "No data received."}

        processed_data = []
        for item in selected_data:
            try:
                # Process each item and include the final product
                processed_data.append({
                    "final_product": final_product,
                    "commodity": item['hs_code'],
                    "origin": item['rule_of_origin'],
                    "description": item.get('description', '')
                })
            except KeyError as e:
                return {"success": False, "error": f"Missing key {e} in item: {item}"}

        if not processed_data:
            return {"success": False, "error": "No valid processed data."}

        # Define JSON file path
        json_file_path = os.path.join(os.getcwd(), 'processed_data.json')
        
        # Write JSON data to a file
        with open(json_file_path, 'w') as json_file:
            json.dump(processed_data, json_file, indent=4)

        # Define Excel file path
        xlsx_file_path = os.path.join(os.getcwd(), 'processed_data.xlsx')

        # Clear the existing Excel file
        if os.path.exists(xlsx_file_path):
            os.remove(xlsx_file_path)

        # Writing to Excel
        import pandas as pd

        df = pd.DataFrame(processed_data)
        df.to_excel(xlsx_file_path, index=False)

        # Return success response with file paths
        return {
            "success": True,
            "processed_data": processed_data,
            "json_file_path": json_file_path,
            "xlsx_file_path": xlsx_file_path
        }

    except Exception as e:
        # Handle any unexpected errors
        return {"success": False, "error": str(e)}
# Function to fetch exchange rates
def get_exchange_rates(base_currency):
    response = requests.get(f"{EXCHANGE_RATE_API_URL}{base_currency}")
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError("Failed to fetch exchange rates.")

# Function to process the uploaded Excel file
def process_excel(file):
    df = pd.read_excel(file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check if all required columns are present
    required_columns = ['item_number', 'description', 'value', 'currency', 'country_of_origin', 'commodity_code']
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
    pdf.set_font("Arial", 'B', 50)
    pdf.set_xy(30, 130)  # Position watermark at the center
    pdf.cell(0, 20, "Tradesphere Global", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)  # Reset text color to black
def generate_beautiful_pdf(data, total, contributions, rates, excel_file='processed_data.xlsx'):
    print("Generating PDF report...")

    # Ensure the output directory existssrc\static src\static
    pdf_output_file = 'src/pdf_report/Enhanced_Summary_Report.pdf'
    os.makedirs(os.path.dirname(pdf_output_file), exist_ok=True)

    # Verify required columns
    required_columns = ['item_number', 'description', 'value_gbp', 'country_of_origin', 'commodity_code', 'currency']
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



    # Title and headerd
    pdf.set_xy(10, 10)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, txt="Summary Report", ln=True, align='C')
    pdf.ln(10)

    # Table Header
    pdf.set_fill_color(200, 200, 200)  # Header color
    pdf.set_font("Arial", 'B', 12)
    col_widths = [20, 55, 25, 40, 40]
    headers = ["Item", "Description", "Value (GBP)", "Country of Origin", "Commodity"]

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
        pdf.ln()

    # Add Summary Section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    #pdf.cell(0, 10, txt="Assembled Place = UK", ln=True)
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
    # Reset text color
    pdf.set_text_color(0, 0, 0) 
    if "MaxNOM" in origin:
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Alternatively, according to MaxNOM Rule:", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Total Value: {total:.2f} GBP", ln=True)
    pdf.cell(0, 10, txt=f"(Calculated using today's exchange rates to convert\n"
                        "non-local currencies into GBP, the currency of the assembled country)", ln=True)
    pdf.ln(5)

    # Contribution Breakdown
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Contribution Breakdown", ln=True)
    #pdf.ln(5)
    pdf.set_font("Arial", size=10)
    for country, percentage in contributions.items():
        pdf.cell(0, 8, txt=f"{country}: {percentage:.2f}%", ln=True)
    pdf.ln(5)  

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Exchange Rates Used:", ln=True)
    pdf.set_font("Arial", size=10)
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



    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, txt=message)

    # Pie Chart for Contributions
    pdf.add_page()
    add_watermark(pdf) 
    pdf.set_font("Arial", 'B', 14)
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
    pdf.set_font("Arial", 'B', 10)
    pdf.multi_cell(0, 8, txt=(
        "Note: Please note that this calculation assumes that all items within the EU/UK "
        "have valid preference origin statements from the suppliers."
    ))
       
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Additionally, you can apply for a binding origin decision at HMRC:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="https://www.gov.uk/guidance/apply-for-a-binding-origin-information-decision", ln=True)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Apply for an Advance Tariff Ruling:", ln=True)
    pdf.set_text_color(0, 0, 255)  # Blue for clickable link
    pdf.set_font("Arial", size=10)

    # Adding a clickable link
    url = "https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling#apply-for-an-advance-tariff-ruling"
    pdf.cell(0, 10, txt="Go to Website", ln=True, link=url)
    
    pdf.set_text_color(0, 0, 0)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, txt=f"Report generated on: {current_datetime}", ln=True)


    # Save PDF
    pdf.output(pdf_output_file)
    print(f"PDF report generated: {pdf_output_file}")
    return pdf_output_file


@app.route('/process-file', methods=['POST'])
def process_file():
    print("Received a file upload request.")
    if 'file' not in request.files:
        print("No file part.")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            print(f"Processing file: {file.filename}")
            df, total_value, contributions, rates = process_excel(file)
            pdf_output_file = generate_beautiful_pdf(df, total_value, contributions, rates)

            # Return the ID or path to allow downloading later
            report_id = os.path.basename(pdf_output_file)
            print(f"PDF generated: {pdf_output_file}")
            return jsonify({'message': 'File processed successfully.', 'download_url': f'/download-report/{report_id}'}), 200

        except KeyError as e:
            missing_columns = ', '.join(e.args)
            print(f"KeyError: Missing columns: {missing_columns}")
            return jsonify({'error': f"Missing required columns: {missing_columns}"}), 400
        except ValueError as e:
            print(f"ValueError: {str(e)}")
            return jsonify({'error': f"Value error: {str(e)}"}), 400
        except FileNotFoundError:
            print(f"FileNotFoundError: The file does not exist or is not accessible.")
            return jsonify({'error': 'File not found'}), 404
        except PermissionError:
            print(f"PermissionError: Insufficient permissions.")
            return jsonify({'error': 'Permission error'}), 403

    print("Unexpected error occurred.")
    return jsonify({'error': 'Unexpected error'}), 500

# Endpoint to serve the generated PDF
@app.route('/download-report/<path:report_id>', methods=['GET'])
def download_report(report_id):
    pdf_dir = 'pdf_report'
    pdf_file_path = os.path.join(pdf_dir, report_id)

    if os.path.exists(pdf_file_path):
        print(f"Serving file: {pdf_file_path}")
        return send_file(pdf_file_path, as_attachment=True, download_name=report_id, mimetype='application/pdf')
    else:
        print(f"File not found: {pdf_file_path}")
        return jsonify({'error': 'File not found'}), 404


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



if __name__ == "__main__":
    app.run(debug=True)
