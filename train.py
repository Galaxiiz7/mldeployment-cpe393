import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")

# Preprocessing
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical variables

X = df.drop("price", axis=1)
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and feature columns
with open("app/model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)
