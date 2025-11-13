import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sys

# ---
# This is a one-time tool to create your 'config.yaml' file.
# ---

# --- STEP 1: Define your users and their plain-text passwords ---
# Add or remove users here as you wish.
users_to_create = {
    "jsmith": {
        "email": "jsmith@example.com",
        "name": "John Smith",
        "password": "abc"  # <-- The plain-text password
    },
    "rdoe": {
        "email": "rdoe@example.com",
        "name": "Rebecca Doe",
        "password": "xyz"  # <-- The plain-text password
    }
}

# --- STEP 2: Hash the passwords (THE CORRECT WAY) ---
hashed_creds = {"usernames": {}}

# Create a single Hasher instance
try:
    hasher_instance = stauth.Hasher()
except Exception as e:
    print(f"Error: Could not instantiate Hasher. Make sure streamlit-authenticator is installed.")
    print(f"Details: {e}")
    sys.exit(1)


for username, details in users_to_create.items():
    # Call the 'hash' method on the instance (this is the correct method name)
    try:
        hashed_password = hasher_instance.hash(details["password"])
    except Exception as e:
        print(f"Error generating hash for user {username}: {e}")
        continue
    
    # Store the user info with the *hashed* password
    hashed_creds["usernames"][username] = {
        "email": details["email"],
        "name": details["name"],
        "password": hashed_password
    }

# --- STEP 3: Define the rest of the config ---
config_data = {
    "credentials": hashed_creds,
    "cookie": {
        "expiry_days": 30,
        "key": "a_random_secret_key_123", # Can be any random string
        "name": "rag_app_cookie" # Can be any random name
    }
    # We correctly omit the 'preauthorized' key
}

# --- STEP 4: Write the config.yaml file automatically ---
try:
    with open('config.yaml', 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)
    print("\n---------------------------------")
    print("✅ SUCCESS!")
    print("Your 'config.yaml' file has been generated.")
    print("You can now run 'streamlit run app.py'")
    print("---------------------------------\n")
except Exception as e:
    print(f"An error occurred while writing config.yaml: {e}")