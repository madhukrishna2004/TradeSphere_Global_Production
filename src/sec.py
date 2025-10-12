import os

# Generate a random 24-byte string
secret_key = os.urandom(24)

# Convert it to a string format if needed
secret_key_str = secret_key.hex()

print(secret_key_str)
