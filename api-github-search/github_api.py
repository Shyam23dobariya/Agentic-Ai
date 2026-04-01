import requests # type: ignore

# ─────────────────────────────────────────────
# GitHub Repository Search API
# ─────────────────────────────────────────────

url = "https://api.github.com/search/repositories"

# Query parameters
params = {
    "q"       : "python",   # Search keyword
    "sort"    : "stars",    # Sort by stars
    "order"   : "desc",     # Descending order
    "per_page": 5           # Limit to 5 results
}

# Make the GET request
response = requests.get(url, params=params)

# Check if request was successful
if response.status_code == 200:
    data = response.json()  # Parse JSON response

    print("=" * 45)
    print("   TOP 5 PYTHON REPOSITORIES ON GITHUB")
    print("=" * 45)

    for i, repo in enumerate(data["items"], start=1):
        name  = repo["full_name"]
        stars = repo["stargazers_count"]
        print(f"\n{i}. Repository : {name}")
        print(f"   Stars      : {stars:,} ⭐")

    print("\n" + "=" * 45)

else:
    print(f"Error: {response.status_code} - {response.text}")