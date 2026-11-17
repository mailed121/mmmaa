import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Load datasets
train = pd.read_csv("house_train.csv")
test = pd.read_csv("house_test.csv")

# Clean training data
train = train.drop_duplicates().dropna(subset=["Price"])

# Separate features and target
X = train.drop(columns=["Price", "ID"])
y = train["Price"]
X_test = test.drop(columns=["ID"])

# Identify columns
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

# Preprocessing
pre = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler()
    ), num),

    ("cat", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ), cat)
])

# Train model
model = make_pipeline(
    pre,
    TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        func=np.log1p,
        inverse_func=np.expm1
    )
)

model.fit(X, y)

# Predict on training data
pred = model.predict(X)

# Evaluate
print("R2 Score:", r2_score(y, pred))

# Plot actual vs predicted
plt.scatter(y, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Polynomial Regression")
plt.show()

# Predict on test data
test_pred = model.predict(X_test)

# Save predictions to CSV
result = pd.DataFrame({
    "ID": test["ID"],
    "Predicted_Price": test_pred
})

result.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")