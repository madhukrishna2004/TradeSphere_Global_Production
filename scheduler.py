from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from db import get_db
from src.routes.preferential_origin import process_and_store_export
 # Adjust path if needed

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
    completed_shift_time = now - timedelta(minutes=1)
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
        print(f"[{datetime.now()}] No files to process for {shift} on {date_str}")
    else:
        for username, filename in files:
            print(f"📦 Auto-processing: {filename} for {username} in {shift}")
            process_and_store_export(username, filename, shift, date_str)

    cur.close()
    conn.close()
