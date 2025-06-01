import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv("housing.csv.xls", delim_whitespace=True, header=None)

# Assign column names (Boston Housing dataset)
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
           "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df.columns = columns

# Split features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
models = {
"Linear Regression": LinearRegression(),
"Ridge (alpha=1.0)": Ridge(alpha=1.0),
"Lasso (alpha=0.1)": Lasso(alpha=0.1)
}

results = {
    "Model": [],
    "R²": [],
    "MSE": [],
    "RMSE": [],
    "MAE": []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n {name}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    results["Model"].append(name)
    results["R²"].append(r2)
    results["MSE"].append(mse)
    results["RMSE"].append(rmse)
    results["MAE"].append(mae)


    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    y_test_np = np.array(y_test)
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Actual vs Predicted Prices: {name}")
    plt.grid(True)
    plt.show()

results_df = pd.DataFrame(results)



metrics = ["R²", "MSE", "RMSE", "MAE"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].bar(results_df["Model"], results_df[metric], color=["skyblue", "red", "lightgreen"])
    axes[idx].set_title(metric)
    axes[idx].set_ylabel(metric)
    axes[idx].set_xticklabels(results_df["Model"], rotation=15)

plt.tight_layout()
plt.suptitle("📊 Model Performance Comparison", fontsize=16, y=1.03)
plt.show()
