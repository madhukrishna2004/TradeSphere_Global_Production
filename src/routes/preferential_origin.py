import os, shutil, zipfile, tempfile
from flask import Blueprint, request, render_template, session, redirect, flash, url_for, send_file
from werkzeug.utils import secure_filename
from datetime import datetime, date
import pandas as pd
from db import get_db
from preferential_folder_setup import get_user_folder_paths, create_user_folders_if_needed
import re
import matplotlib.pyplot as plt

preferential_origin_bp = Blueprint("preferential_origin", __name__)

# 🔁 Determine shift based on current time
'''def get_current_shift():
    now = datetime.now().time()
    if now <= datetime.strptime("06:00", "%H:%M").time():
        return "Shift 1"
    elif now <= datetime.strptime("12:00", "%H:%M").time():
        return "Shift 2"
    elif now <= datetime.strptime("18:00", "%H:%M").time():
        return "Shift 3"
    else:
        return "Shift 4"'''

# 🔁 Simulate PDF generation
def generate_dummy_pdf_report(bom_path, pdf_output_path):
    with open(pdf_output_path, "w") as f:
        f.write(f"PDF Report for {os.path.basename(bom_path)}")

# 🔁 Generate abstract Excel for a shift
def create_shift_abstract(username, shift, export_date, export_files, abstract_path):
    df = pd.DataFrame(export_files, columns=["BoM File", "Export Type", "Report Path"])
    df["Shift"] = shift
    df["Username"] = username
    df["Date"] = export_date
    df.to_excel(abstract_path, index=False)

from datetime import datetime, time

import os
 
from fpdf import FPDF
import pandas as pd
from db import get_db

def process_and_store_export(username, filename, shift, export_date):
    base_server, base_local = get_user_folder_paths(username)
    
    bom_path = os.path.join(base_local, "BOM_Inputs", filename)

    # PDF & Abstract output folders
    pdf_dir_eu = os.path.join(base_local, "PDFReports", "EU_Preferential")
    abstract_dir = os.path.join(base_local, "AbstractReports")
    os.makedirs(pdf_dir_eu, exist_ok=True)
    os.makedirs(abstract_dir, exist_ok=True)

    # Create dummy PDF
    pdf_file_name = filename.replace(".xlsx", "_YES.pdf")
    pdf_path = os.path.join(pdf_dir_eu, pdf_file_name)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Dummy PDF Export for {filename}", ln=True)
    pdf.output(pdf_path)

    # Save to export_files
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO export_files (username, export_date, shift, export_type, file_name, file_path, export_timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (username, export_date, shift, 'EU', pdf_file_name, pdf_path, datetime.now()))
    conn.commit()

    # Also generate abstract (dummy summary)
    abstract_data = {
        "Processed File": [filename],
        "PDF Export": [pdf_file_name],
        "Status": ["Preferential"]
    }
    df = pd.DataFrame(abstract_data)

    abstract_file_name = f"{export_date}_{shift.replace(' ', '_')}_Abstract.xlsx"
    abstract_path = os.path.join(abstract_dir, abstract_file_name)
    df.to_excel(abstract_path, index=False)

    # Save abstract file to export_files (optional)
    cur.execute("""
        INSERT INTO export_files (username, export_date, shift, export_type, file_name, file_path, export_timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (username, export_date, shift, 'ABSTRACT', abstract_file_name, abstract_path, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()

from datetime import datetime, time

def get_current_shift():
    now = datetime.now().time()
    if time(0, 0) <= now <= time(6, 0):
        return 'Shift 1'
    elif time(6, 1) <= now <= time(12, 0):
        return 'Shift 2'
    elif time(12, 1) <= now <= time(18, 0):
        return 'Shift 3'
    else:
        return 'Shift 4'

from flask import Markup, url_for

from flask import Markup, url_for
 
@preferential_origin_bp.route("/upload", methods=["GET", "POST"])
def upload_bom():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]
    create_user_folders_if_needed(username)

    if request.method == "POST":
        files = request.files.getlist("bom_file")
        if not files:
            flash("No files selected.")
            return redirect(url_for("preferential_origin.upload_bom"))

        shift = get_current_shift()
        upload_date = datetime.now().date()

        conn = get_db()
        cur = conn.cursor()

        for file in files:
            if file and file.filename.endswith(".xlsx"):
                filename = secure_filename(file.filename)
                base_server, base_local = get_user_folder_paths(username)

                local_path = os.path.join(base_local, "BOM_Inputs", filename)
                server_path = os.path.join(base_server, "BOM_Inputs", filename)

                file.save(local_path)
                shutil.copy(local_path, server_path)

                # Save initial file info in DB
                cur.execute("""
                    INSERT INTO bom_uploads (username, original_filename, saved_path, server_path, shift, upload_date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (username, filename, local_path, server_path, shift, upload_date))
                bom_id = cur.fetchone()[0]

                try:
                    df = pd.read_excel(local_path)

                    conversion_data = {
                        "USD": 0.78,
                        "SEK": 0.074,
                        "PLN": 0.19,
                        "GBP": 1.0,
                        "EUR": 0.85
                    }

                    clean_name = os.path.splitext(filename)[0]

                    # ✅ Generate both reports and capture paths
                    eu_path = generate_uk_eu_report(df, conversion_data, clean_name)
                    jp_path = generate_uk_japan_report(df, conversion_data, clean_name)

                    # ✅ Store report paths only
                    cur.execute("""
                        UPDATE bom_uploads
                        SET uk_eu_report_path = %s,
                            uk_japan_report_path = %s
                        WHERE id = %s
                    """, (eu_path, jp_path, bom_id))

                    print(f"[✔] Reports saved: {eu_path} and {jp_path}")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

        conn.commit()
        cur.close()
        conn.close()

        flash("BoM files uploaded and reports generated successfully.")
        return redirect(url_for("preferential_origin.upload_bom"))

    return render_template("preferential_origin_upload.html")

from flask import send_file
import urllib.parse

@preferential_origin_bp.route("/view-report")
def view_report():
    if "user" not in session:
        return redirect("/login")

    raw_path = request.args.get("path")
    if not raw_path:
        return "Invalid file path", 400

    # Decode the URL-encoded path
    decoded_path = urllib.parse.unquote(raw_path)

    # Construct absolute path from current working dir
    abs_path = os.path.abspath(os.path.join(os.getcwd(), decoded_path))

    # Optional security check (ensure inside `originreports`)
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "originreports"))
    if not abs_path.startswith(base_dir):
        return "Access denied", 403

    if not os.path.isfile(abs_path):
        return f"File not found: {abs_path}", 404

    return send_file(abs_path, mimetype="application/pdf", as_attachment=False)

@preferential_origin_bp.route("/exports", methods=["GET"])
def show_exports():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]
    selected_date = request.args.get("date") or date.today().strftime("%Y-%m-%d")

    conn = get_db()
    cur = conn.cursor()
    
    # Fetch from bom_uploads
    cur.execute("""
        SELECT shift, original_filename, uk_eu_report_path, uk_japan_report_path
        FROM bom_uploads
        WHERE username = %s AND upload_date = %s
        ORDER BY shift, id DESC
    """, (username, selected_date))
    
    files = []
    for row in cur.fetchall():
        shift, original_filename, eu_path, jp_path = row
        if eu_path:
            files.append({
                "shift": shift,
                "export_type": "UK–EU Report",
                "file_name": f"{original_filename} – UK–EU",
                "file_path": eu_path
            })
        if jp_path:
            files.append({
                "shift": shift,
                "export_type": "UK–Japan Report",
                "file_name": f"{original_filename} – UK–Japan",
                "file_path": jp_path
            })

    cur.close()
    conn.close()

    # Abstract reports, if any
    base_local = get_user_folder_paths(username)[1]
    abstracts = {}
    for shift in ["Shift 1", "Shift 2", "Shift 3", "Shift 4"]:
        path = os.path.join(base_local, "AbstractReports", f"Abstract_{shift}_{selected_date}.xlsx")
        if os.path.exists(path):
            abstracts[shift] = path

    return render_template("exports_dashboard.html", files=files, date=selected_date, abstracts=abstracts)

# ✅ File Download
@preferential_origin_bp.route("/download-file")
def download_file():
    path = request.args.get("file_path")
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

# ✅ Download shift files ZIP
@preferential_origin_bp.route("/download-shift", methods=["POST"])
def download_shift():
    if "user" not in session:
        return redirect("/login")
    
    shift = request.form.get("shift")
    date_selected = request.form.get("date")
    username = session["user"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_name, file_path FROM export_files
        WHERE username = %s AND export_date = %s AND shift = %s
    """, (username, date_selected, shift))
    files = cur.fetchall()
    cur.close()
    conn.close()

    if not files:
        return "No files found", 404

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zipf:
            for fname, fpath in files:
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=fname)
        return send_file(tmp.name, as_attachment=True, download_name=f"{date_selected}_{shift}_Exports.zip")

# ✅ Download all day ZIP
@preferential_origin_bp.route("/download-day", methods=["POST"])
def download_day():
    if "user" not in session:
        return redirect("/login")

    date_selected = request.form.get("date")
    username = session["user"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_name, file_path FROM export_files
        WHERE username = %s AND export_date = %s
    """, (username, date_selected))
    files = cur.fetchall()
    cur.close()
    conn.close()

    if not files:
        return "No files for selected date", 404

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zipf:
            for fname, fpath in files:
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=fname)
        return send_file(tmp.name, as_attachment=True, download_name=f"{date_selected}_All_Shifts_Exports.zip")
@preferential_origin_bp.route("/trigger-abstract", methods=["POST"])
def trigger_shift_abstract():
    if "user" not in session:
        return redirect("/login")

    shift = request.form.get("shift")
    export_date = request.form.get("date")
    username = session["user"]

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT file_name, export_type, file_path FROM export_files
        WHERE username = %s AND export_date = %s AND shift = %s
    """, (username, export_date, shift))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        flash(f"⚠️ No export files found for {shift} on {export_date}.")
        return redirect(url_for("preferential_origin.show_exports", date=export_date))

    export_records = [[row[0], row[1], row[2]] for row in rows]
    base_local = get_user_folder_paths(username)[1]
    abstract_path = os.path.join(base_local, "AbstractReports", f"Abstract_{shift}_{export_date}.xlsx")
    create_shift_abstract(username, shift, export_date, export_records, abstract_path)

    flash(f"✅ Abstract for {shift} regenerated successfully.")
    return redirect(url_for("preferential_origin.show_exports", date=export_date))


from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from db import get_db
#from export_generator import process_and_store_export  # Make sure this function exists and is imported

def get_shift_name_from_time(current_time):
    hour = current_time.hour
    minute = current_time.minute
    if hour < 6 or (hour == 6 and minute == 0):
        return 'Shift 1'
    elif hour < 12 or (hour == 12 and minute == 0):
        return 'Shift 2'
    elif hour < 18 or (hour == 18 and minute == 0):
        return 'Shift 3'
    else:
        return 'Shift 4'

def process_completed_shift():
    now = datetime.now()
    completed_shift_time = now - timedelta(minutes=1)  # Small buffer
    shift = get_shift_name_from_time(completed_shift_time)

    date_str = completed_shift_time.date()

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT username, original_filename
        FROM bom_uploads
        WHERE shift = %s AND upload_date = %s
    """, (shift, date_str))

    files = cur.fetchall()
    if not files:
        print(f"[{datetime.now()}] No files for {shift} on {date_str}")
    else:
        print(f"[{datetime.now()}] Auto-processing {len(files)} files for {shift}")
        for username, filename in files:
            process_and_store_export(username, filename, shift, date_str)

    cur.close()
    conn.close()

import requests

EXCHANGE_RATE_API_URL = "https://api.exchangerate-api.com/v4/latest/"  # Replace with your actual base

# 1. Fetch exchange rates
def get_exchange_rates(base_currency):
    response = requests.get(f"{EXCHANGE_RATE_API_URL}{base_currency}")
    if response.status_code == 200:
        return response.json().get("rates", {})
    else:
        raise ValueError("Failed to fetch exchange rates.")

# 2. Convert amount from one currency to another
def convert_currency(amount, from_currency, to_currency="GBP"):
    if from_currency == to_currency:
        return amount  # No conversion needed

    try:
        rates = get_exchange_rates(from_currency)
        rate = rates.get(to_currency)
        if not rate:
            raise ValueError(f"No exchange rate found for {to_currency}")
        return round(amount * rate, 2)
    except Exception as e:
        print(f"❌ Currency conversion failed: {e}")
        return None

 
import pandas as pd

 
    
 
import pandas as pd
from fpdf import FPDF
import os
import pandas as pd

def parse_uploaded_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        required_columns = [
            "Item Number", "Description", "Value", "Currency", "COO",
            "Preference", "CC", "SAV", "SAC",
            "SA COO", "SAP", "SA HS Code",
            "FCC", "Weight (kg)"
        ]
        # Ensure all required columns are present
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"❌ Missing columns in Excel: {', '.join(missing)}")

        return df
    except Exception as e:
        print(f"❌ Error parsing Excel: {e}")
        return None

# Ensure folders exist
os.makedirs("originreports/uk_eu", exist_ok=True)
os.makedirs("originreports/uk_japan", exist_ok=True)

 

from fpdf import FPDF
import os

def convert_to_gbp(value, currency, conversion_dict):
    try:
        if pd.isna(value) or pd.isna(currency):
            return ""
        rate = conversion_dict.get(currency.upper(), 1)
        return round(float(value) * rate, 2)
    except:
        return ""

import os
from fpdf import FPDF

def safe_text(text):
    """
    Safely normalize text for PDF: replaces problematic Unicode characters.
    """
    if isinstance(text, str):
        return (
            text.replace("–", "-")   # en dash
                .replace("—", "-")   # em dash
                .replace("’", "'")   # right apostrophe
                .replace("‘", "'")   # left apostrophe
                .replace("“", '"')   # left double quote
                .replace("”", '"')   # right double quote
                .replace("\xa0", " ")  # non-breaking space
                .strip()
        )
    return str(text)

from fpdf import FPDF
import os
import pandas as pd

def safe_text(text):
    if pd.isna(text):
        return ""
    if isinstance(text, str):
        return text.replace("–", "-").replace("’", "'").strip()
    return str(int(text)) if isinstance(text, float) and text.is_integer() else str(text)

class PDFWithFooter(FPDF):
    def header(self):
        self.add_font('DejaVu', '', 'src/static/assets/fonts/DejaVuSans.ttf', uni=True)
        self.set_font("DejaVu", '', 40)
        self.set_text_color(230, 230, 230)

        # Calculate text width for horizontal centering
        watermark_text = "TradeSphere Global"
        text_width = self.get_string_width(watermark_text)
        page_width = self.w
        page_height = self.h

        x = (page_width - text_width) / 2
        y = page_height / 2

        self.text(x=x, y=y, txt=watermark_text)

        # Reset text color to black
        self.set_text_color(0, 0, 0)


    def footer(self):
        self.set_y(-15)
        self.add_font('DejaVu', '', 'src/static/assets/fonts/DejaVuSans.ttf', uni=True)
        self.set_font("DejaVu", '', 8)
        self.set_text_color(0)
        self.cell(0, 5, f"TradeSphere Global | Krislynx LLP | 2025", 0, 1, 'C')
        self.cell(0, 5, f"Page {self.page_no()}", 0, 0, 'C')

def sanitize_commodity_code(code: str) -> str:
    """
    Sanitize a commodity code by removing non-digit characters and preserving valid 8 or 10 digit format.
    """
    if not code:
        return ""

    code = str(code).strip().replace('.', '').replace(' ', '').replace('-', '')
    code = ''.join(filter(str.isdigit, code))

    # Allow only 8 or 10 digit codes — fallback to original if invalid
    if len(code) in [8, 10]:
        return code
    elif len(code) < 8:
        # Optionally pad with trailing zeros to reach 8 digits
        return code.ljust(8, '0')
    elif 8 < len(code) < 10:
        # If 9 digits — treat it as malformed, fallback to 8
        return code[:8]
    else:
        return code[:10]

def check_uk_eu_preference(final_commodity_code: str) -> str:
    """
    Get UK–Japan product-specific rule of origin.
    Cleanly handles exact and fallback lookups (10/8/6/4 digits) 
    without corrupting the commodity code (e.g., avoids leading zero issues).
    """
    try:
        import pandas as pd

        df = pd.read_excel("global-uk-tariff.xlsx", dtype=str)

        # Step 1: Clean and validate input
        code_raw = final_commodity_code.strip().replace('.', '')

        # Remove accidental leading zeros (but NOT zeros inside the code)
        code_raw = code_raw.lstrip('0')

        if not code_raw.isdigit():
            return None  # invalid format

        # Pad back correctly if needed (never adding leading zeros!)
        fallback_codes = []

        code_len = len(code_raw)
        if code_len in [10, 8, 6, 4]:
            fallback_codes.append(code_raw)
            if code_len < 10:
                fallback_codes.append(code_raw.ljust(10, '0'))  # trailing only
        elif code_len < 4:
            return None  # too short to be valid
        else:
            # Unknown length, fallback safely
            fallback_codes.append(code_raw[:10])
            fallback_codes.append(code_raw[:8])
            fallback_codes.append(code_raw[:6])
            fallback_codes.append(code_raw[:4])

        # Deduplicate while keeping order
        seen = set()
        fallback_codes = [x for x in fallback_codes if not (x in seen or seen.add(x))]

        # Step 3: Match from top to bottom
        for code in fallback_codes:
            matched = df[df['commodity'].astype(str).str.strip() == code]
            if not matched.empty:
                rule_col = 'Product-specific rule of origin'
                if rule_col in matched.columns:
                    rule = matched.iloc[0][rule_col]
                    if isinstance(rule, str) and rule.strip():
                        return rule.strip()

        return None  # No valid match

    except Exception as e:
        print(f"⚠️ Japan rule lookup failed for code {final_commodity_code}: {e}")
        return None


from datetime import datetime
import uuid
from fpdf import FPDF
from datetime import datetime
import os, uuid
from textwrap import wrap

def generate_uk_eu_report(df, conversion_dict, origin_lookup):
    os.makedirs("originreports/uk_eu", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_code = uuid.uuid4().hex[:6].upper()
    filename = f"report_{timestamp}_{unique_code}_uk_eu.pdf"
    out_path = os.path.join("originreports/uk_eu", filename)

    pdf = PDFWithFooter(orientation='L')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_font('DejaVu', '', 'src/static/assets/fonts/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, "UK–EU Origin Report", ln=True, align='C')
    pdf.ln(5)

    df = df[df["Item Number"].notna() & (df["Item Number"].astype(str).str.strip() != "")]

    # --- Country Contribution Calculations ---
    total_value = df['Value'].apply(lambda v: convert_to_gbp(v, df.loc[df['Value'] == v, 'Currency'].values[0], conversion_dict)).sum()

    # Grouped contribution by Country of Origin
    contributions = (
        df.groupby('COO')['Value']
        .apply(lambda s: sum([convert_to_gbp(v, df.loc[df['Value'] == v, 'Currency'].values[0], conversion_dict) for v in s]))
        .apply(lambda x: (x / total_value) * 100 if total_value else 0)
        .round(2)
        .to_dict()
)


    excluded_cols = ["Currency", "SAC"]
    display_cols = [col for col in df.columns if col not in excluded_cols]

    display_headers = []
    for col in display_cols:
        if col == "Value":
            display_headers.append("Value (GBP)")
        elif col == "SAV":
            display_headers.append("SAV (GBP)")
        else:
            display_headers.append(safe_text(col))

    pdf.set_font("DejaVu", '', 9)
    table_width = pdf.w * 0.95
    col_width = table_width / len(display_cols)
    line_height = 6

    x_start = pdf.get_x()
    y_start = pdf.get_y()

    for i, header in enumerate(display_headers):
        pdf.set_xy(x_start + i * col_width, y_start)
        pdf.multi_cell(col_width, line_height, header, border=1, align='C')

    pdf.set_xy(x_start, y_start + line_height)

    for _, row in df.iterrows():
        x = x_start
        y = pdf.get_y()
        max_lines = 1
        row_cells = []

        for col in display_cols:
            val = row[col]
            if col == "Value":
                val = convert_to_gbp(row["Value"], row["Currency"], conversion_dict)
            elif col == "SAV":
                val = convert_to_gbp(row["SAV"], row["SAC"], conversion_dict)
            text = safe_text(val)
            row_cells.append(text)
            lines = max(1, int(len(text) / (col_width / 2)))
            max_lines = max(max_lines, lines)

        row_height = max_lines * line_height

        for i, cell in enumerate(row_cells):
            pdf.set_xy(x + i * col_width, y)
            pdf.multi_cell(col_width, line_height, cell, border=1, align='L')

        pdf.set_y(y + row_height)

    # Final Commodity Codes Section
    pdf.ln(5)
    pdf.set_font("DejaVu", '', 10)
    final_codes = df["FCC"].dropna().unique()
    clean_codes = ", ".join(safe_text(code) for code in final_codes)
    pdf.cell(0, 10, "Final Commodity Code(s): " + clean_codes, ln=True)

    # Rule Lookup Section
    seen_codes = set()
    pdf.ln(3)
    pdf.set_font("DejaVu", '', 10)

    final_code_rules = {}  # 🔁 Store rule per code for later use

    for code in final_codes:
        clean_code = sanitize_commodity_code(str(code).strip())
        if clean_code in seen_codes or not clean_code:
            continue
        seen_codes.add(clean_code)

        try:
            rule = check_uk_eu_preference(clean_code)
        except Exception as e:
            rule = f"⚠️ Rule lookup failed: {str(e)}"

        # 🔐 Store for later use
        final_code_rules[clean_code] = rule

        # 📜 Immediate rule print section
        rule_text = f"• Rule of Origin for {clean_code}: {safe_text(rule)}"
        wrapped_rule = "\n".join(wrap(rule_text, width=100))
        pdf.multi_cell(pdf.w - 20, 8, wrapped_rule, border=0)


        # Interpret Rules
        #rule = rule.upper()
    for code, rule in final_code_rules.items():
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        if 'CTH' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CTH: CTH means production from non-originating materials of any heading, "
                "except that of the product. This means that any non-originating material used must be classified "
                "under a heading (4-digit level of the Harmonised System) other than the final product (i.e., a change in heading). "
                "Since the commodity codes in the bill of materials differ from the first four digits of the final product's code, "
                "this product qualifies for UK–Japan Preferential Duty (Zero or reduced rate)."
            ))

        elif 'CTSH' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CTSH: Production must use materials from a different subheading "
                "(6-digit level). Since BOM codes differ from the first six digits of the final product's code, "
                "this product qualifies under the UK–Japan Preference for reduced or zero import duty."
            ))

        elif 'CC' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CC: Production from non-originating materials must involve a change in Chapter (2-digit level). "
                "Since the BOM components are from different Chapters, this product qualifies under the trade preference rule."
            ))

        # Extra rule clarifications
        if "cathode" in rule.lower():
            pdf.ln(5)
            pdf.set_font("ARIAL", size=9)
            pdf.cell(0, 10, txt="*Ensure no active cathode materials are present to qualify under the CTH rule.", ln=True)

        if "MaxNOM" in rule:
            pdf.ln(5)
            pdf.set_font("ARIAL", size=9)
            pdf.cell(0, 10, txt="Alternatively, this product may qualify under the MaxNOM Rule.", ln=True)
        
    pdf.set_font("DejaVu", '', 11)  # Reset font if changed

    # Abbreviations Section
    pdf.ln(8)
    pdf.set_font("DejaVu", '', 10)
    pdf.cell(0, 8, "Abbreviations:", ln=True)
    abbrev_text = (
        "COO = Country Of Origin\n"
        "CC = Commodity Code\n"
        "SAV = Sub Assembly Value\n"
        "SAC = Sub Assembly Currency\n"
        "SA COO = Sub Assembly COO\n"
        "SAP = Sub Assembly Preference\n"
        "SA HS Code = Sub Assembly HS Code\n"
        "FCC = Final Commodity Code"
    )
    for line in abbrev_text.split("\n"):
        pdf.cell(0, 6, line, ln=True)

    # --- Country Contributions Section ---
    pdf.ln(8)
    pdf.set_font("DejaVu", '', 10)
    pdf.cell(0, 8, "Country Contribution (% of total value):", ln=True)

    if contributions:
        for country, pct in contributions.items():
            line = f"- {country}: {pct:.2f}%"
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "No contribution data available.", ln=True)
        

    # --- Contribution Analysis ---
    pdf.ln(6)
    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Country Contribution Analysis:", ln=True)
    pdf.set_font("ARIAL", size=10)

    eu_countries = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
    ]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]

    # Total % from UK and eu
    uk_eu_percentage = sum(
        percent for country, percent in contributions.items()
        if country.strip() in eu_countries + uk_countries
    )

    # Rest of world %
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country.strip() not in eu_countries + uk_countries
    )

    # Get highest contributing non-UK/Japan country
    filtered_contributions = {
        country: percent for country, percent in contributions.items()
        if country.strip() not in eu_countries + uk_countries
    }
    highest_contributed_country = max(filtered_contributions, key=filtered_contributions.get, default="Unknown")

    # Display %
    pdf.cell(0, 8, txt=f"Total UK & EU Contribution: {uk_eu_percentage:.2f}%", ln=True)
    pdf.cell(0, 8, txt=f"Rest of World Contribution: {rest_percentage:.2f}%", ln=True)
    pdf.cell(0, 8, txt=f"Highest Non-UK/EU Contributor: {highest_contributed_country} ({filtered_contributions.get(highest_contributed_country, 0):.2f}%)", ln=True)

    # Optional logic usage for further origin rule checks
    max_nom_percentage = rest_percentage  # For rule logic if needed
        

    # Eligibility Analysis Summary Section (Final PDF Content)
    pdf.add_page()
    pdf.set_font("DejaVu", '', 10)
     
    pdf.ln(5)

    for clean_code in final_code_rules:
        origin = final_code_rules.get(clean_code, "")
        message = ""

         
        # Rule Check Logic
        if "wholly obtained" in origin.lower():
            message = (
                f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                " The product is eligible under the UK-EU Preference trade agreement for zero or reduced duty while importing."
            )

        elif "MaxNOM" in origin:
            match = re.search(r"MaxNOM\s*(\d+)\s?%", origin)
            if match:
                threshold = int(match.group(1))
                if max_nom_percentage < threshold:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        " The product is eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                    )
                else:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        "The product is not eligible under the UK-EU Preference trade agreement for zero or reduced duty while importing."
                    )
            else:
                message = " Invalid MaxNOM condition specified."

         

        else:
            if max_nom_percentage < 50:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    "The product is eligible under the UK-EU Preference trade agreement for zero or reduced duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    "The product is not eligible under the UK-EU Preference trade agreement for zero or reduced duty while importing."
                )

        # Final Message Section
        #pdf.set_font("DejaVu", '', 10)
        #pdf.multi_cell(0, 5, message)
        #pdf.ln(4)


    # Ensure this font is registered, or use 'Arial' which FPDF2 allows if available in system
    pdf.set_font("Arial", 'I', 11)

    # Instead of ❌ or symbols not supported by Helvetica/Arial, use plain text
    clean_message = message.replace("❌", "Error:")  # Or just remove the symbol
    pdf.multi_cell(0, 8, txt=clean_message)

     

    # Pie Chart for Contributions
    pdf.add_page()

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Pie Chart of Contributions", ln=True)
    pdf.ln(5)

    eu_countries = [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
    ]
    uk_countries = [
        "England", "Scotland", "Wales", "Northern Ireland",
        "UK", "United Kingdom", "Great Britain"
    ]

    # Calculate contributions
    uk_eu_percentage = sum(
        percent for country, percent in contributions.items()
        if country in eu_countries or country in uk_countries
    )
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in eu_countries and country not in uk_countries
    )

    labels = ['Other Countries', 'UK & EU']
    sizes = [rest_percentage, uk_eu_percentage]
    colors = ['#FF9999', '#66B2FF']

    # Save pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, wedgeprops=dict(edgecolor='black')
    )
    plt.axis('equal')
    plt.title('Contribution Breakdown', fontsize=14)

    # Make sure the folder exists
    
    os.makedirs('pdf_report', exist_ok=True)

    pie_chart_path = 'pdf_report/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    pdf.image(pie_chart_path, x=50, y=30, w=180)

    # Final Page with Notes and Links
    pdf.add_page()
    pdf.ln(10)

    # Red Note
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", 'B', 10)
    pdf.multi_cell(0, 8, txt=(
        "Note: This calculation assumes that all items within the UK/EU "
        "have valid preference origin statements from their suppliers."
    ))

    # Blue Link: Binding Origin Decision
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Apply for a binding origin decision (HMRC):", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0, 10,
        txt="https://www.gov.uk/guidance/apply-for-a-binding-origin-information-decision",
        ln=True
    )

    # Blue Link: Advance Tariff Ruling
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Apply for an Advance Tariff Ruling:", ln=True)
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0, 10,
        txt="Go to Website",
        ln=True,
        link="https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling#apply-for-an-advance-tariff-ruling"
    )

    # Footer: Generation Timestamp
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'I', 10)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, txt=f"Report generated on: {current_datetime}", ln=True)

    pdf.output(out_path)
    print(f"[✔] UK–EU report saved: {out_path}")
    return out_path


def check_uk_japan_preference(final_commodity_code: str) -> str:
    """
    Get UK–Japan product-specific rule of origin.
    Cleanly handles exact and fallback lookups (10/8/6/4 digits) 
    without corrupting the commodity code (e.g., avoids leading zero issues).
    """
    try:
        import pandas as pd

        df = pd.read_excel("global-uk-tariff.xlsx", dtype=str)

        # Step 1: Clean and validate input
        code_raw = final_commodity_code.strip().replace('.', '')

        # Remove accidental leading zeros (but NOT zeros inside the code)
        code_raw = code_raw.lstrip('0')

        if not code_raw.isdigit():
            return None  # invalid format

        # Pad back correctly if needed (never adding leading zeros!)
        fallback_codes = []

        code_len = len(code_raw)
        if code_len in [10, 8, 6, 4]:
            fallback_codes.append(code_raw)
            if code_len < 10:
                fallback_codes.append(code_raw.ljust(10, '0'))  # trailing only
        elif code_len < 4:
            return None  # too short to be valid
        else:
            # Unknown length, fallback safely
            fallback_codes.append(code_raw[:10])
            fallback_codes.append(code_raw[:8])
            fallback_codes.append(code_raw[:6])
            fallback_codes.append(code_raw[:4])

        # Deduplicate while keeping order
        seen = set()
        fallback_codes = [x for x in fallback_codes if not (x in seen or seen.add(x))]

        # Step 3: Match from top to bottom
        for code in fallback_codes:
            matched = df[df['commodity'].astype(str).str.strip() == code]
            if not matched.empty:
                rule_col = 'Product-specific rule of origin japan'
                if rule_col in matched.columns:
                    rule = matched.iloc[0][rule_col]
                    if isinstance(rule, str) and rule.strip():
                        return rule.strip()

        return None  # No valid match

    except Exception as e:
        print(f"⚠️ Japan rule lookup failed for code {final_commodity_code}: {e}")
        return None

from datetime import datetime
import uuid
from fpdf import FPDF
from datetime import datetime
import os, uuid
from textwrap import wrap

def generate_uk_japan_report(df, conversion_dict, origin_lookup):
    os.makedirs("originreports/uk_japan", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_code = uuid.uuid4().hex[:6].upper()
    filename = f"report_{timestamp}_{unique_code}_uk_japan.pdf"
    out_path = os.path.join("originreports/uk_japan", filename)

    pdf = PDFWithFooter(orientation='L')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_font('DejaVu', '', 'src/static/assets/fonts/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, "UK–Japan Origin Report", ln=True, align='C')
    pdf.ln(5)

    df = df[df["Item Number"].notna() & (df["Item Number"].astype(str).str.strip() != "")]

    # --- Country Contribution Calculations ---
    total_value = df['Value'].apply(lambda v: convert_to_gbp(v, df.loc[df['Value'] == v, 'Currency'].values[0], conversion_dict)).sum()

    # Grouped contribution by Country of Origin
    contributions = (
        df.groupby('COO')['Value']
        .apply(lambda s: sum([convert_to_gbp(v, df.loc[df['Value'] == v, 'Currency'].values[0], conversion_dict) for v in s]))
        .apply(lambda x: (x / total_value) * 100 if total_value else 0)
        .round(2)
        .to_dict()
)


    excluded_cols = ["Currency", "SAC"]
    display_cols = [col for col in df.columns if col not in excluded_cols]

    display_headers = []
    for col in display_cols:
        if col == "Value":
            display_headers.append("Value (GBP)")
        elif col == "SAV":
            display_headers.append("SAV (GBP)")
        else:
            display_headers.append(safe_text(col))

    pdf.set_font("DejaVu", '', 9)
    table_width = pdf.w * 0.95
    col_width = table_width / len(display_cols)
    line_height = 6

    x_start = pdf.get_x()
    y_start = pdf.get_y()

    for i, header in enumerate(display_headers):
        pdf.set_xy(x_start + i * col_width, y_start)
        pdf.multi_cell(col_width, line_height, header, border=1, align='C')

    pdf.set_xy(x_start, y_start + line_height)

    for _, row in df.iterrows():
        x = x_start
        y = pdf.get_y()
        max_lines = 1
        row_cells = []

        for col in display_cols:
            val = row[col]
            if col == "Value":
                val = convert_to_gbp(row["Value"], row["Currency"], conversion_dict)
            elif col == "Sub Assembly Value":
                val = convert_to_gbp(row["Sub Assembly Value"], row["Sub Assembly Currency"], conversion_dict)
            text = safe_text(val)
            row_cells.append(text)
            lines = max(1, int(len(text) / (col_width / 2)))
            max_lines = max(max_lines, lines)

        row_height = max_lines * line_height

        for i, cell in enumerate(row_cells):
            pdf.set_xy(x + i * col_width, y)
            pdf.multi_cell(col_width, line_height, cell, border=1, align='L')

        pdf.set_y(y + row_height)

    # Final Commodity Codes Section
    pdf.ln(5)
    pdf.set_font("DejaVu", '', 10)
    final_codes = df["FCC"].dropna().unique()
    clean_codes = ", ".join(safe_text(code) for code in final_codes)
    pdf.cell(0, 10, "Final Commodity Code(s): " + clean_codes, ln=True)

    # Rule Lookup Section
    seen_codes = set()
    pdf.ln(3)
    pdf.set_font("DejaVu", '', 10)

    final_code_rules = {}  # 🔁 Store rule per code for later use

    for code in final_codes:
        clean_code = sanitize_commodity_code(str(code).strip())
        if clean_code in seen_codes or not clean_code:
            continue
        seen_codes.add(clean_code)

        try:
            rule = check_uk_japan_preference(clean_code)
        except Exception as e:
            rule = f"⚠️ Rule lookup failed: {str(e)}"

        # 🔐 Store for later use
        final_code_rules[clean_code] = rule

        # 📜 Immediate rule print section
        rule_text = f"• Rule of Origin for {clean_code}: {safe_text(rule)}"
        wrapped_rule = "\n".join(wrap(rule_text, width=100))
        pdf.multi_cell(pdf.w - 20, 8, wrapped_rule, border=0)


        # Interpret Rules
        #rule = rule.upper()
    for code, rule in final_code_rules.items():
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        if 'CTH' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CTH: CTH means production from non-originating materials of any heading, "
                "except that of the product. This means that any non-originating material used must be classified "
                "under a heading (4-digit level of the Harmonised System) other than the final product (i.e., a change in heading). "
                "Since the commodity codes in the bill of materials differ from the first four digits of the final product's code, "
                "this product qualifies for UK–Japan Preferential Duty (Zero or reduced rate)."
            ))

        elif 'CTSH' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CTSH: Production must use materials from a different subheading "
                "(6-digit level). Since BOM codes differ from the first six digits of the final product's code, "
                "this product qualifies under the UK–Japan Preference for reduced or zero import duty."
            ))

        elif 'CC' in rule:
            pdf.multi_cell(0, 8, txt=(
                "According to CC: Production from non-originating materials must involve a change in Chapter (2-digit level). "
                "Since the BOM components are from different Chapters, this product qualifies under the trade preference rule."
            ))

        # Extra rule clarifications
        if "cathode" in rule.lower():
            pdf.ln(5)
            pdf.set_font("ARIAL", size=9)
            pdf.cell(0, 10, txt="*Ensure no active cathode materials are present to qualify under the CTH rule.", ln=True)

        if "MaxNOM" in rule:
            pdf.ln(5)
            pdf.set_font("ARIAL", size=9)
            pdf.cell(0, 10, txt="Alternatively, this product may qualify under the MaxNOM Rule.", ln=True)
        
    pdf.set_font("DejaVu", '', 11)  # Reset font if changed

    # Abbreviations Section
    pdf.ln(8)
    pdf.set_font("DejaVu", '', 10)
    pdf.cell(0, 8, "Abbreviations:", ln=True)
    abbrev_text = (
        "COO = Country Of Origin\n"
        "CC = Commodity Code\n"
        "SAV = Sub Assembly Value\n"
        "SAC = Sub Assembly Currency\n"
        "SA COO = Sub Assembly COO\n"
        "SAP = Sub Assembly Preference\n"
        "SA HS Code = Sub Assembly HS Code\n"
        "FCC = Final Commodity Code"
    )
    for line in abbrev_text.split("\n"):
        pdf.cell(0, 6, line, ln=True)

    # --- Country Contributions Section ---
    pdf.ln(8)
    pdf.set_font("DejaVu", '', 10)
    pdf.cell(0, 8, "Country Contribution (% of total value):", ln=True)

    if contributions:
        for country, pct in contributions.items():
            line = f"- {country}: {pct:.2f}%"
            pdf.cell(0, 6, line, ln=True)
    else:
        pdf.cell(0, 6, "No contribution data available.", ln=True)
        

    # --- Contribution Analysis ---
    pdf.ln(6)
    pdf.set_font("ARIAL", size=12)
    pdf.cell(0, 10, txt="Country Contribution Analysis:", ln=True)
    pdf.set_font("ARIAL", size=10)

    japan_countries = ["Japan"]
    uk_countries = ["England", "Scotland", "Wales", "Northern Ireland", "UK", "United Kingdom", "Great Britain"]

    # Total % from UK and Japan
    uk_japan_percentage = sum(
        percent for country, percent in contributions.items()
        if country.strip() in japan_countries + uk_countries
    )

    # Rest of world %
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country.strip() not in japan_countries + uk_countries
    )

    # Get highest contributing non-UK/Japan country
    filtered_contributions = {
        country: percent for country, percent in contributions.items()
        if country.strip() not in japan_countries + uk_countries
    }
    highest_contributed_country = max(filtered_contributions, key=filtered_contributions.get, default="Unknown")

    # Display %
    pdf.cell(0, 8, txt=f"Total UK & Japan Contribution: {uk_japan_percentage:.2f}%", ln=True)
    pdf.cell(0, 8, txt=f"Rest of World Contribution: {rest_percentage:.2f}%", ln=True)
    pdf.cell(0, 8, txt=f"Highest Non-UK/Japan Contributor: {highest_contributed_country} ({filtered_contributions.get(highest_contributed_country, 0):.2f}%)", ln=True)

    # Optional logic usage for further origin rule checks
    max_nom_percentage = rest_percentage  # For rule logic if needed
        

    # Eligibility Analysis Summary Section (Final PDF Content)
    pdf.add_page()
    pdf.set_font("DejaVu", '', 10)
     
    pdf.ln(5)

    for clean_code in final_code_rules:
        origin = final_code_rules.get(clean_code, "")
        message = ""

         
        # Rule Check Logic
        if "wholly obtained" in origin.lower():
            message = (
                f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                " The product is eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
            )

        elif "MaxNOM" in origin:
            match = re.search(r"MaxNOM\s*(\d+)\s?%", origin)
            if match:
                threshold = int(match.group(1))
                if max_nom_percentage < threshold:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        " The product is eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                    )
                else:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        " The product is not eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                    )
            else:
                message = " Invalid MaxNOM condition specified."

        elif "RVC" in origin:
            match = re.search(r"RVC\s*(\d+)\s?%\s?\(FOB\)", origin)
            if match:
                threshold = int(match.group(1))
                originating_percentage = sum(
                    percent for country, percent in contributions.items()
                    if country in japan_countries or country in uk_countries
                )
                rvc_percentage = (originating_percentage / 100) * total_value / total_value * 100

                if rvc_percentage >= threshold:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        " The product is eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                    )
                else:
                    message = (
                        f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                        " The product is not eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                    )
            else:
                message = " Invalid RVC condition specified."

        else:
            if max_nom_percentage < 50:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    "✅ The product is eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                )
            else:
                message = (
                    f"Based on the findings, according to the product-specific rule of origin of the final product: {origin}.\n"
                    " The product is not eligible under the UK-Japan Preference trade agreement for zero or reduced duty while importing."
                )

        # Final Message Section
        #pdf.set_font("DejaVu", '', 10)
        #pdf.multi_cell(0, 5, message)
        #pdf.ln(4)


    # Ensure this font is registered, or use 'Arial' which FPDF2 allows if available in system
    pdf.set_font("Arial", 'I', 11)

    # Instead of ❌ or symbols not supported by Helvetica/Arial, use plain text
    clean_message = message.replace("❌", "Error:")  # Or just remove the symbol
    pdf.multi_cell(0, 8, txt=clean_message)

     

    # Pie Chart for Contributions
    pdf.add_page()

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Pie Chart of Contributions", ln=True)
    pdf.ln(5)

    japan_countries = ["Japan"]
    uk_countries = [
        "England", "Scotland", "Wales", "Northern Ireland",
        "UK", "United Kingdom", "Great Britain"
    ]

    # Calculate contributions
    uk_japan_percentage = sum(
        percent for country, percent in contributions.items()
        if country in japan_countries or country in uk_countries
    )
    rest_percentage = sum(
        percent for country, percent in contributions.items()
        if country not in japan_countries and country not in uk_countries
    )

    labels = ['Other Countries', 'UK & Japan']
    sizes = [rest_percentage, uk_japan_percentage]
    colors = ['#FF9999', '#66B2FF']

    # Save pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, wedgeprops=dict(edgecolor='black')
    )
    plt.axis('equal')
    plt.title('Contribution Breakdown', fontsize=14)

    # Make sure the folder exists
    
    os.makedirs('pdf_report', exist_ok=True)

    pie_chart_path = 'pdf_report/pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    pdf.image(pie_chart_path, x=50, y=30, w=180)

    # Final Page with Notes and Links
    pdf.add_page()
    pdf.ln(10)

    # Red Note
    pdf.set_text_color(255, 0, 0)
    pdf.set_font("Arial", 'B', 10)
    pdf.multi_cell(0, 8, txt=(
        "Note: This calculation assumes that all items within the UK/Japan "
        "have valid preference origin statements from their suppliers."
    ))

    # Blue Link: Binding Origin Decision
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Apply for a binding origin decision (HMRC):", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0, 10,
        txt="https://www.gov.uk/guidance/apply-for-a-binding-origin-information-decision",
        ln=True
    )

    # Blue Link: Advance Tariff Ruling
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, txt="Apply for an Advance Tariff Ruling:", ln=True)
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0, 10,
        txt="Go to Website",
        ln=True,
        link="https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling#apply-for-an-advance-tariff-ruling"
    )

    # Footer: Generation Timestamp
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'I', 10)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, txt=f"Report generated on: {current_datetime}", ln=True)

    pdf.output(out_path)
    print(f"[✔] UK–Japan report saved: {out_path}")
    return out_path
