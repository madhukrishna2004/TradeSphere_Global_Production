import os

def get_user_folder_paths(username):
    user_home = os.path.expanduser("~")
    base_local = os.path.join(user_home, "TradeSphere Global", username)
    base_server = os.path.join("secure_server", username)
    return base_server, base_local

def create_user_folders_if_needed(username):
    base_server, base_local = get_user_folder_paths(username)
    folders = [
        "BOM_Inputs",
        "PDFReports/EU_Preferential",
        "PDFReports/JP_Preferential",
        "AbstractReports",
        "ExportTemplate"
    ]
    for folder in folders:
        server_path = os.path.join(base_server, folder)
        local_path = os.path.join(base_local, folder)

        os.makedirs(server_path, exist_ok=True)
        os.makedirs(local_path, exist_ok=True)

        print(f"[✓] Server Path Created: {server_path}")
        print(f"[✓] Local Path Created: {local_path}")

    return base_server, base_local
