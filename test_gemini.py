import os
import google.generativeai as genai
import toml

print("--- Starting Gemini API Test (using secrets.toml) ---")

# 1. Define the path to your secrets file
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
api_key = None

# 2. Load the API key from the secrets.toml file
try:
    secrets = toml.load(secrets_file_path)
    api_key = secrets.get("GEMINI_API_KEY")
    if not api_key:
        print(f"❌ ERROR: 'GEMINI_API_KEY' not found inside {secrets_file_path}.")
    else:
        print("✅ API Key loaded successfully from secrets.toml.")

except FileNotFoundError:
    print(f"❌ ERROR: The secrets file was not found at {secrets_file_path}.")
except Exception as e:
    print(f"❌ ERROR: An error occurred while reading the secrets file: {e}")


if api_key:
    try:
        # 3. Configure the Gemini client
        genai.configure(api_key=api_key)
        
        # 4. Create a simple prompt with the corrected model name
        # --- THIS IS THE FIX ---
        model = genai.GenerativeModel('gemini-pro')
        # ----------------------
        
        prompt = "In one simple sentence, what is an exoplanet?"
        print(f"\nSending a test prompt to Gemini: '{prompt}'")
        
        # 5. Generate a response
        response = model.generate_content(prompt)
        
        # 6. Print the result
        print("\n✅ SUCCESS! Response from Gemini:")
        print(f"-> {response.text}")

    except Exception as e:
        print(f"\n❌ ERROR: An error occurred while contacting the API.")
        print(f"   Details: {e}")

print("\n--- Test complete ---")