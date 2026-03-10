import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
# Note: The SOCR dataset often has spaces in column names like " Height(Inches)"
df = pd.read_csv("heights.csv")

# Clean column names (removes leading/trailing spaces)
df.columns = df.columns.str.strip()

# Drop Index if it exists
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

# 2. Define Features and Target
# Ensure these match the exact column names in your CSV
X = df[['Height(Inches)']] 
y = df['Weight(Pounds)']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Visualizations (The "GitHub-Ready" Plots)
plt.figure(figsize=(12, 5))

# Plot 1: Regression Line
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Height vs Weight: Regression Line')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.legend()

# Plot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Weight')
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')

plt.tight_layout()
plt.show()
