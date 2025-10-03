import os
import google.generativeai as genai
import toml

print("--- Checking Available Gemini Models ---")

# 1. Load the API key from the secrets.toml file
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
api_key = None

try:
    secrets = toml.load(secrets_file_path)
    api_key = secrets.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ ERROR: 'GEMINI_API_KEY' not found inside {secrets_file_path}.")
    else:
        print("✅ API Key loaded successfully.")
        genai.configure(api_key=api_key)

except FileNotFoundError:
    print(f"❌ ERROR: The secrets file was not found at {secrets_file_path}.")
except Exception as e:
    print(f"❌ ERROR: An error occurred while reading the secrets file: {e}")


if api_key:
    try:
        print("\n--- Models available to your API key ---")
        for model in genai.list_models():
            # We are checking which models support the 'generateContent' method
            if 'generateContent' in model.supported_generation_methods:
                print(model.name)
    except Exception as e:
        print(f"\n❌ ERROR: Could not retrieve model list.")
        print(f"   Details: {e}")

print("\n--- Check complete ---")