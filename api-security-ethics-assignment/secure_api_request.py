import os
import requests # type: ignore

# STEP 1: Retrieve API key securely from environment variable
api_key = os.environ.get("API_KEY")

if not api_key:
    print("Error: API_KEY environment variable not set.")
    print("Please set it using:")
    print("  Windows PowerShell : $env:API_KEY='my_secret_key_123'")
    
    exit(1)

# STEP 2: Define the API endpoint
# url = "https://api.example.com/data"
url = "https://httpbin.org/bearer"

# STEP 3: Include API key in request headers (Authorization Bearer format)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type" : "application/json"
}

# STEP 4: Send GET request and handle status codes
try:
    response = requests.get(url, headers=headers)

    # ── Status Code Handling ──────────────────────────────────────
    if response.status_code == 200:
        print(" Success! JSON Response:")
        print(response.json())

    elif response.status_code == 429:
        print("⚠️  Rate limit reached. Try again later.")

    else:
        print(f" Request failed with status code: {response.status_code}")

except requests.exceptions.ConnectionError:
    print(" Connection Error: Could not reach the API endpoint.")
    print("   Note: 'api.example.com' is a placeholder URL for demonstration.")

except requests.exceptions.Timeout:
    print(" Request timed out. Please try again.")

except requests.exceptions.RequestException as e:
    print(f" An unexpected error occurred: {e}")