import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# FETCH & CLEAN DATA
# This file handles:
#   1. Fetching data from the API
#   2. Converting JSON to DataFrame
#   3. Cleaning the data
#   4. Adding post_length column
#   5. Groupby analysis
# ─────────────────────────────────────────────────────────────────

def fetch_data():
    """Fetch post data from JSONPlaceholder API."""
    url = "https://jsonplaceholder.typicode.com/posts"
    response = requests.get(url)
    data = response.json()
    return data


def clean_data(data):
    """Convert JSON to DataFrame and perform basic cleaning."""

    # STEP 2: Convert JSON to Pandas DataFrame
    df = pd.DataFrame(data)

    # STEP 3: Basic Cleaning
    df = df.rename(columns={"userId": "user_id"})  # Rename userId → user_id
    df = df.drop(columns=["id"])                    # Drop the id column

    # STEP 5: Create new column — post_length
    df["post_length"] = df["body"].apply(len)

    return df


def get_posts_per_user(df):
    """STEP 4: Count how many posts each user created using groupby()."""
    posts_per_user = df.groupby("user_id")["title"].count().reset_index()
    posts_per_user.columns = ["user_id", "post_count"]
    return posts_per_user


# ── Run standalone to verify data ────────────────────────────────
if __name__ == "__main__":
    print("Fetching data from API...")
    raw_data = fetch_data()
    print(f"✅ Fetched {len(raw_data)} records.")

    print("\nCleaning data...")
    df = clean_data(raw_data)
    print(f"✅ Cleaned DataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    print("\nPosts per user:")
    posts_per_user = get_posts_per_user(df)
    print(posts_per_user)

    print("\nPost length stats:")
    print(df["post_length"].describe())