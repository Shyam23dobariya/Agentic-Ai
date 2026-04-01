import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────────────────────────────
# STEP 1: Load the Iris Dataset
# ─────────────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Observation: Dataset loaded successfully with 150 rows and 5 columns.
# It contains measurements for 3 species: setosa, versicolor, virginica.

print("=" * 55)
print("         IRIS DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────
# STEP 2: Inspect the Dataset Structure
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 2: Dataset Structure ──")
print(f"Shape       : {df.shape}")          # rows x columns
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")

# Observation: 4 numeric features (float64) and 1 categorical column (species).
# No unexpected data types detected.

# ─────────────────────────────────────────────────────────────────
# STEP 3: Column Info and Missing Values
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 3: Column Info & Missing Values ──")
print(df.info())
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nSummary Statistics:\n{df.describe()}")

# Observation: No missing values found in any column.
# All 150 records are complete — no data cleaning required.
# Each species has exactly 50 samples (balanced dataset).

# ─────────────────────────────────────────────────────────────────
# STEP 4: Distribution of Petal Length (Histogram)
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 4: Petal Length Distribution ──")

fig_hist = px.histogram(
    df,
    x="petal_length",
    color="species",
    nbins=20,
    barmode="overlay",
    opacity=0.75,
    title="Distribution of Petal Length by Species",
    labels={"petal_length": "Petal Length (cm)", "count": "Frequency"},
    color_discrete_map={
        "setosa": "#636EFA",
        "versicolor": "#EF553B",
        "virginica": "#00CC96"
    }
)
fig_hist.update_layout(
    xaxis_title="Petal Length (cm)",
    yaxis_title="Count",
    legend_title="Species",
    width=800, height=500
)
fig_hist.show()

# Observation: Setosa has very short petal lengths (1–2 cm), clearly separated
# from versicolor (3–5 cm) and virginica (4.5–7 cm).
# This makes petal_length a strong feature for species classification.

# ─────────────────────────────────────────────────────────────────
# STEP 5: Outlier Detection using Box Plots
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 5: Outlier Detection (Box Plots) ──")

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

fig_box = px.box(
    df,
    x="species",
    y=df[features[0]],   # initial; loop below generates all 4
    color="species",
    title="Box Plot — All Features by Species",
    color_discrete_map={
        "setosa": "#636EFA",
        "versicolor": "#EF553B",
        "virginica": "#00CC96"
    }
)

# Generate a separate box plot for each feature
for feature in features:
    fig = px.box(
        df,
        x="species",
        y=feature,
        color="species",
        points="all",          # show all data points (outliers highlighted)
        title=f"Box Plot of {feature.replace('_', ' ').title()} by Species",
        labels={feature: f"{feature.replace('_', ' ').title()} (cm)"},
        color_discrete_map={
            "setosa": "#636EFA",
            "versicolor": "#EF553B",
            "virginica": "#00CC96"
        }
    )
    fig.update_layout(width=800, height=500)
    fig.show()

# Observation: sepal_width shows a few potential outliers in setosa (very wide).
# Petal features have minimal outliers — they are highly consistent within species.

# ─────────────────────────────────────────────────────────────────
# STEP 6: Relationship Between Variables (Scatter Plot)
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 6: Petal Length vs Petal Width ──")

fig_scatter = px.scatter(
    df,
    x="petal_length",
    y="petal_width",
    color="species",
    symbol="species",
    title="Petal Length vs Petal Width by Species",
    labels={
        "petal_length": "Petal Length (cm)",
        "petal_width" : "Petal Width (cm)"
    },
    color_discrete_map={
        "setosa": "#636EFA",
        "versicolor": "#EF553B",
        "virginica": "#00CC96"
    },
    trendline="ols"           # Ordinary Least Squares trendline per species
)
fig_scatter.update_layout(width=800, height=500)
fig_scatter.show()

# Observation: Strong positive correlation between petal_length and petal_width.
# Setosa clusters tightly at the bottom-left (small petals).
# Versicolor and Virginica overlap slightly but are still separable.

# ─────────────────────────────────────────────────────────────────
# STEP 7: Species Insights — Pair Plot (all feature combinations)
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 7: Species Insights — Pair Plot ──")

fig_pair = px.scatter_matrix(
    df,
    dimensions=features,
    color="species",
    title="Pair Plot — All Feature Combinations by Species",
    color_discrete_map={
        "setosa": "#636EFA",
        "versicolor": "#EF553B",
        "virginica": "#00CC96"
    }
)
fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
fig_pair.update_layout(width=900, height=900)
fig_pair.show()

# Observation: Petal features (length & width) provide the clearest species
# separation. Setosa is linearly separable from the other two species.
# Versicolor and Virginica overlap more, especially in sepal dimensions.

# ─────────────────────────────────────────────────────────────────
# STEP 8: Species Count — Verify Balance
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 8: Species Count ──")
print(df["species"].value_counts())

fig_bar = px.bar(
    df["species"].value_counts().reset_index(),
    x="species",
    y="count",
    color="species",
    title="Sample Count per Species",
    labels={"species": "Species", "count": "Count"},
    color_discrete_map={
        "setosa": "#636EFA",
        "versicolor": "#EF553B",
        "virginica": "#00CC96"
    }
)
fig_bar.update_layout(width=600, height=400, showlegend=False)
fig_bar.show()

# Observation: Dataset is perfectly balanced — 50 samples per species.
# No class imbalance issue for machine learning model training.

# ─────────────────────────────────────────────────────────────────
# STEP 9: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────
print("\n── STEP 9: Correlation Heatmap ──")

corr = df[features].corr().round(2)
print(f"\nCorrelation Matrix:\n{corr}")

fig_heat = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Heatmap",
    zmin=-1, zmax=1
)
fig_heat.update_layout(width=600, height=500)
fig_heat.show()

# Observation: petal_length and petal_width are highly correlated (r ≈ 0.96).
# sepal_width has a weak or negative correlation with petal features.
# This suggests petal dimensions carry similar information (possible redundancy).

# ─────────────────────────────────────────────────────────────────
# EDA SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("                    EDA SUMMARY")
print("=" * 55)
print("""
1. Dataset    : 150 rows × 5 columns, no missing values.
2. Balance    : Perfectly balanced — 50 samples per species.
3. Best Feature: petal_length best separates the 3 species.
4. Outliers   : Minor outliers in sepal_width (setosa).
5. Correlation: petal_length & petal_width highly correlated.
6. Separability:
   - Setosa    → linearly separable (small petals).
   - Versicolor & Virginica → slight overlap in sepal space.
7. ML Readiness: Dataset is clean and ready for model training.
""")