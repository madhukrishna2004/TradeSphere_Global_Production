from cryptography.fernet import Fernet

KEY_FILE = "encryption.key"
ENCRYPTED_KEY_FILE = "encrypted_api_key_parts.txt"

# --- Utility functions ---
def generate_encryption_key():
    return Fernet.generate_key()

def save_encryption_key(key):
    with open(KEY_FILE, "wb") as f:
        f.write(key)

def load_encryption_key():
    with open(KEY_FILE, "rb") as f:
        return f.read()

def encrypt_api_key_parts(part1, part2, key):
    cipher = Fernet(key)
    encrypted_part1 = cipher.encrypt(part1.encode())
    encrypted_part2 = cipher.encrypt(part2.encode())
    with open(ENCRYPTED_KEY_FILE, "wb") as f:
        f.write(encrypted_part1 + b"\n" + encrypted_part2)
    print(f"Encrypted API key parts saved to {ENCRYPTED_KEY_FILE}")


# --- Run once to generate key + encrypted parts ---
if __name__ == "__main__":
    # Replace with your actual parts
    part1 = "sk-proj-rk7Pl7JS_Xzm4QHLVvyUJO2DAUiMT50fvX1Y6AgJP4MD27-sGTA7ofa-fp_n_u6WmOa"
    part2 = "nIUB4sCT3BlbkFJJb5z9BAZNGjfbI21K08S9tYQjMUhoZX4gwHlJYLQEtuayeBUn4wMClKIOwelMMtSgNbCsNyuUA"

    # Generate and save key
    encryption_key = generate_encryption_key()
    save_encryption_key(encryption_key)

    # Encrypt and save key parts
    encrypt_api_key_parts(part1, part2, encryption_key)
