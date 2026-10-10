import pandas as pd

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display the shape of the dataset
print("\nDataset shape:")
print(df.shape)

# Display the column names
print("\nColumn names:")
print(df.columns.tolist())
# Check basic information about the dataset
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check the number of diabetic and non-diabetic cases
print("\nOutcome Distribution:")
print(df["Outcome"].value_counts())
# Replace unrealistic zero values with NaN
columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[columns_with_zero] = df[columns_with_zero].replace(0, pd.NA)

# Check the newly identified missing values
print("\nMissing values after replacing unrealistic zeros:")
print(df.isnull().sum())
# Fill missing values with the median of each column
for column in columns_with_zero:
    df[column] = df[column].fillna(df[column].median())

# Check again for missing values
print("\nMissing values after median imputation:")
print(df.isnull().sum())
# Separate features and target variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Fit the scaler only on training data
X_train = scaler.fit_transform(X_train)

# Use the same scaler to transform testing data
X_test = scaler.transform(X_test)

print("\nFeature scaling completed successfully.")
print("Scaled training data shape:", X_train.shape)
print("Scaled testing data shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)

# Train the model using the training data
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredictions made successfully.")
print("First 10 predictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 2))
print("Recall   :", round(recall, 2))
print("F1 Score :", round(f1, 2))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
# Plot the distribution of diabetes outcomes
plt.figure(figsize=(6, 5))

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of People")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])

plt.show()
# Plot distribution of important features
plt.figure(figsize=(8, 5))

sns.histplot(df["Glucose"], kde=True)

plt.title("Glucose Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Number of People")

plt.show()
# Plot BMI distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["BMI"], kde=True)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Number of People")

plt.show()
# Plot Age distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["Age"], kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")

plt.show()
# Correlation heatmap
plt.figure(figsize=(10, 8))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Diabetes Dataset")

plt.show()
# Compare actual and predicted outcomes
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted values:")
print(comparison.head(20))
import joblib

# Save the trained model and scaler
joblib.dump(model, "diabetes_logistic_model.pkl")
joblib.dump(scaler, "diabetes_scaler.pkl")

print("\nModel and scaler saved successfully.")
# Test the model with a new person's data

new_person = [[
    2,       # Pregnancies
    120,     # Glucose
    70,      # BloodPressure
    20,      # SkinThickness
    79,      # Insulin
    25.0,    # BMI
    0.5,     # DiabetesPedigreeFunction
    30       # Age
]]

# Scale the new person's data
new_person_scaled = scaler.transform(new_person)

# Make prediction
prediction = model.predict(new_person_scaled)

print("\nNew Person Prediction:")

if prediction[0] == 1:
    print("The model predicts: Diabetes")
else:
    print("The model predicts: No Diabetes")
    # Interactive diabetes prediction

print("\n--- Diabetes Prediction System ---")

pregnancies = float(input("Enter number of pregnancies: "))
glucose = float(input("Enter glucose level: "))
blood_pressure = float(input("Enter blood pressure: "))
skin_thickness = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
diabetes_pedigree = float(input("Enter Diabetes Pedigree Function: "))
age = float(input("Enter age: "))

user_data = pd.DataFrame([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]], columns=X.columns)

user_data_scaled = scaler.transform(user_data)

user_prediction = model.predict(user_data_scaled)

print("\nPrediction Result:")

if user_prediction[0] == 1:
    print("The model predicts: Diabetes")
else:
    print("The model predicts: No Diabetes")