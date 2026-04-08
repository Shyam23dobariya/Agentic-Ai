import plotly.express as px
import streamlit as st
from fetch_data import fetch_data, clean_data, get_posts_per_user

# ─────────────────────────────────────────────────────────────────
# STREAMLIT DASHBOARD
# Run with: streamlit run dashboard.py
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Post Data Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Simple Data Dashboard")
st.markdown("Fetching and analyzing post data from JSONPlaceholder API")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Fetch & Clean Data (using fetch_data.py)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw_data = fetch_data()
    df = clean_data(raw_data)
    return df

with st.spinner("Fetching data from API..."):
    df = load_data()

posts_per_user = get_posts_per_user(df)

st.success(f"✅ Data fetched successfully! Total records: {len(df)}")

# ─────────────────────────────────────────────────────────────────
# SECTION 1: Dataset Preview
# ─────────────────────────────────────────────────────────────────
st.subheader("📋 Dataset Preview")
st.markdown("First 10 rows of the cleaned dataset:")
st.dataframe(df.head(10), use_container_width=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("📄 Total Posts",        len(df))
col2.metric("👤 Total Users",        df["user_id"].nunique())
col3.metric("📝 Avg Post Length",    f"{df['post_length'].mean():.0f} chars")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SECTION 2: Posts Per User — Bar Chart
# ─────────────────────────────────────────────────────────────────
st.subheader("👤 Posts Per User")
st.markdown("Number of posts created by each user:")

fig_bar = px.bar(
    posts_per_user,
    x="user_id",
    y="post_count",
    color="post_count",
    color_continuous_scale="Blues",
    title="Posts Per User",
    labels={
        "user_id"   : "User ID",
        "post_count": "Number of Posts"
    },
    text="post_count"
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(
    xaxis=dict(tickmode="linear"),
    coloraxis_showscale=False,
    height=450
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SECTION 3: Post Length Distribution — Histogram
# ─────────────────────────────────────────────────────────────────
st.subheader("📏 Post Length Distribution")
st.markdown("Distribution of character count across all posts:")

fig_hist = px.histogram(
    df,
    x="post_length",
    nbins=20,
    color_discrete_sequence=["#636EFA"],
    title="Distribution of Post Length",
    labels={"post_length": "Post Length (characters)"}
)
fig_hist.update_layout(
    xaxis_title="Post Length (characters)",
    yaxis_title="Number of Posts",
    height=450
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SECTION 4: Summary Statistics
# ─────────────────────────────────────────────────────────────────
st.subheader("📊 Post Length Summary Statistics")
st.dataframe(
    df["post_length"].describe().reset_index().rename(
        columns={"index": "Statistic", "post_length": "Value"}
    ),
    use_container_width=True
)

st.markdown("---")
st.caption("Data source: https://jsonplaceholder.typicode.com/posts")