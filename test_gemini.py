# --- test_gemini.py ---
import google.generativeai as genai
import key_param # Assuming your key is in key_param.py
import os

print("--- Starting Direct Gemini API Test ---")

try:
    print(f"Using API Key from key_param.py")
    # Ensure your key_param.gemini_api_key is correctly loaded
    if not hasattr(key_param, 'gemini_api_key') or not key_param.gemini_api_key:
         raise ValueError("gemini_api_key not found or is empty in key_param.py")

    genai.configure(api_key=key_param.gemini_api_key)

    print("\nListing available models accessible by this API key...")
    # List models to see what's available
    available_models = []
    for m in genai.list_models():
        # Check if the model supports the 'generateContent' method needed
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name} (Supports generateContent)")
            available_models.append(m.name)
        else:
            print(f"  - {m.name} (Does NOT support generateContent)")

    # Check if the target model is listed and supported
    target_model = 'gemini-1.0-pro'
    print(f"\nChecking availability of '{target_model}'...")
    if target_model in available_models:
        print(f"'{target_model}' seems available and supports generateContent.")
    else:
        print(f"!!! WARNING: '{target_model}' was NOT found in the list of available models supporting generateContent for this API key.")
        # You might want to try a different model from the list above if this happens

    print(f"\nAttempting to use '{target_model}' directly...")
    # Use the specific model name
    model = genai.GenerativeModel(target_model)
    response = model.generate_content("Explain what a large language model is in one sentence.")

    print("\n--- Direct API Test Response ---")
    # Check response content safely
    if hasattr(response, 'text'):
        print(response.text)
    else:
        print("Response received, but 'text' attribute not found.")
        print("Full Response:", response)

    print("\n--- Direct API Test Successful! ---")

except Exception as e:
    print(f"\n!!! Direct API Test FAILED: {e}")
    print("Check: API Key validity, Project Billing Status, Network Connection, Model Name.")

print("\n--- Test Finished ---")