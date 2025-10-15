from fastapi import FastAPI
from ap_msme_one import fetch_profile, apply_schemes

app = FastAPI()

@app.post("/ap-msme-sync")
async def sync_msme(msme_id: str):
    profile = await fetch_profile(msme_id)  # AP MSME ONE API
    readiness = calculate_readiness(profile, ap_datasets)  # Google Drive datasets
    docs = generate_ap_docs(profile)  # GST invoices, e-CoO
    incentives = apply_schemes(profile)  # Auto-apply 5% AP export subsidy
    return {"score": readiness, "docs": docs, "incentives": incentives}
