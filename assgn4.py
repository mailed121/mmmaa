import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("drug(1).csv")

X = data.drop("Drug", axis=1)
y = data["Drug"]

# Convert categorical columns
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction and accuracy
y_pred = model.predict(X_test)

print("Predicted Drugs:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Draw Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns,
          class_names=model.classes_,
          filled=True, rounded=True)
plt.show()