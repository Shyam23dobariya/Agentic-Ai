import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # ==========================================
    # DATA PREPARATION
    # ==========================================
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Train-test split (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (Fit ONLY on training data, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # ==========================================
    # PART 1 — Build a Metric Comparison Table
    # ==========================================
    print("--- Part 1: Metric Comparison Table ---")
    metrics_results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics_results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }

    # Convert to DataFrame and set index name to "Model"
    df_metrics = pd.DataFrame.from_dict(metrics_results, orient='index')
    df_metrics.index.name = "Model"
    print(df_metrics)
    print("\n")

    # ==========================================
    # PART 2 — Overfitting Check on the Decision Tree
    # ==========================================
    print("--- Part 2: Overfitting Check on Decision Tree ---")
    dt_model = models["Decision Tree"]
    
    # Calculate train and test accuracies
    dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train_scaled))
    dt_test_acc = df_metrics.loc["Decision Tree", "Accuracy"]
    gap = dt_train_acc - dt_test_acc

    print(f"Training Accuracy: {dt_train_acc:.4f}")
    print(f"Test Accuracy:     {dt_test_acc:.4f}")
    print(f"Train-Test Gap:    {gap:.4f}")

    # One-line diagnostic logic
    if gap > 0.05:
        print("⚠ Overfitting detected — gap exceeds 5%")
    elif dt_train_acc < 0.85:
        print("⚠ Underfitting detected — training accuracy too low")
    else:
        print("✓ Model is at the sweet spot — acceptable gap")
    print("\n")

    # ==========================================
    # PART 3 — Apply the "Start Simple, Go Complex" Protocol
    # ==========================================
    print("--- Part 3: 'Start Simple, Go Complex' Protocol ---")
    
    # Best F1-Score across all models
    best_f1 = max(df_metrics["F1-Score"])
    selected_model_name = None

    # Order is already defined in the dictionary (LR -> DT -> RF)
    for name in models.keys():
        current_f1 = df_metrics.loc[name, "F1-Score"]
        # Check if within 2 percentage points (0.02) of the best score
        if (best_f1 - current_f1) <= 0.02:
            selected_model_name = name
            break  # Stop at the simplest model meeting the condition

    selected_f1 = df_metrics.loc[selected_model_name, "F1-Score"]
    print(f"Selected Model: {selected_model_name} | F1-Score: {selected_f1:.4f}\n")

    # ==========================================
    # PART 4 — Save the Selected Model
    # ==========================================
    print("--- Part 4: Save the Selected Model ---")
    model_filename = "tumour_classifier_v1.joblib"
    scaler_filename = "tumour_scaler_v1.joblib"

    # Save selected model and scaler
    joblib.dump(models[selected_model_name], model_filename)
    print(f"✓ Model saved as {model_filename}")
    
    joblib.dump(scaler, scaler_filename)
    print(f"✓ Scaler saved as {scaler_filename}\n")

    # ==========================================
    # PART 5 — Verify Model Persistence
    # ==========================================
    print("--- Part 5: Verify Model Persistence ---")
    # Load files back from disk
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)

    # Use loaded scaler to transform original unscaled X_test
    X_test_loaded_scaled = loaded_scaler.transform(X_test)

    # Generate predictions using the loaded model
    loaded_preds = loaded_model.predict(X_test_loaded_scaled)

    # Print first 10 predictions and actual labels
    print("Predictions vs Actual (first 10):")
    print(f"Predicted: {list(loaded_preds[:10])}")
    print(f"Actual:    {list(y_test[:10])}")

if __name__ == "__main__":
    main()