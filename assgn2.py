import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def run_regression(name, file, feature, target):

    # Load dataset
    df = pd.read_csv(file)
    X = df[[feature]]
    y = df[target]

    # Split dataset (70:30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and MSE
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Training:", len(X_train), "Testing:", len(X_test))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Save actual and predicted values to CSV
    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    result.to_csv(name.replace(" ", "_") + "_results.csv", index=False)
    print("Results saved successfully!")

    # Regression graph
    sorted_X = X.sort_values(feature)

    plt.scatter(X, y, label="Actual Data")
    plt.plot(sorted_X, model.predict(sorted_X),
             color="red", label="Regression Line")

    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(name + " - Linear Regression")
    plt.legend()
    plt.grid()
    plt.savefig(name.replace(" ", "_") + "_graph.png")
    plt.show()


# Run for all three datasets
experiments = [
    ("Fuel Consumption", "fuel_consumption_dataset.csv",
     "ENGINESIZE", "CO2EMISSIONS"),

    ("Used Cars", "Used_Car_Dataset.csv",
     "Year", "Price_INR"),

    ("China GDP", "LR_or_PR_china_gdp.csv",
     "Year", "Value")
]

for name, file, feature, target in experiments:
    try:
        run_regression(name, file, feature, target)
    except FileNotFoundError:
        print("File not found:", file)