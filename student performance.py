
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "student": ["A", "B", "C", "D", "E"],
    "study_hours": [2, 4, 3, 6, 8],
    "attendance": [60, 70, 75, 90, 95],
    "marks": [40, 55, 50, 75, 90]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

print("\nAverage Marks:")
print(df["marks"].mean())

print("\nHighest Marks:")
print(df["marks"].max())

print("\nLowest Marks:")
print(df["marks"].min())


print("\nAverage Study Hours:")
print(df["study_hours"].mean())

print("\nAverage Attendance:")
print(df["attendance"].mean())

print("\nCorrelation between Study Hours and Marks:")
print(df["study_hours"].corr(df["marks"]))

print("\nCorrelation between Attendance and Marks:")
print(df["attendance"].corr(df["marks"]))


print("\nStudents Who Need Improvement:")

low_performers = df[df["marks"] < 50]

print(low_performers)
plt.bar(df["student"], df["marks"])

#missing value
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#handling missing value
print("\nTesting Missing Values:")
test_df = df.copy()
test_df.loc[0, "marks"] = None
print(test_df)
print("\nMissing Values:")
print(test_df.isnull().sum())

#filling missing value
print("\nBefore Filling Missing Values:")
print(test_df["marks"])
average_marks = test_df["marks"].mean()
test_df["marks"] = test_df["marks"].fillna(average_marks)
print("\nAfter Filling Missing Values:")
print(test_df["marks"])
print("\nRemaining Missing Values:")
print(test_df["marks"].isnull().sum())

#Analyse Attendance Performance
plt.figure(figsize=(8, 5))
plt.scatter(
    df["attendance"],
    df["marks"],
    s=100,
    alpha=0.8
)
for i in range(len(df)):
    plt.annotate(
        df["student"][i],
        (df["attendance"][i], df["marks"][i]),
        xytext=(5, 5),
        textcoords="offset points"
    )
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("Attendance vs Student Marks")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Relationship b/w attendance and marks
correlation = df["attendance"].corr(df["marks"])
print("Attendance-Marks Correlation:", correlation)
if correlation > 0:
    print("Finding: Positive relationship")
elif correlation < 0:
    print("Finding: Negative relationship")
else:
    print("Finding: No linear correlation")

#ML Model (Linear Regression)
X = df[["study_hours"]]
y = df["marks"]
model = LinearRegression()
model.fit(X, y)
predicted_marks = model.predict(pd.DataFrame({"study_hours":[5]}))
print("Predicted marks for 5 study hours:",
      predicted_marks[0])

#Evaluate the predict models
from sklearn.metrics import mean_absolute_error
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
print("Actual Marks:")
print(list(y))
print("\nPredicted Marks:")
print(predictions)
print("\nMean Absolute Error:", mae)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#Testing case
# Train using training data only
model = LinearRegression()
model.fit(X_train, y_train)
# Predict marks for unseen test data
y_pred = model.predict(X_test)
print("Actual test marks:")
print(list(y_test))
print("\nPredicted test marks:")
print(y_pred)
# Evaluate test predictions
mae = mean_absolute_error(y_test, y_pred)
print("\nTest MAE:", mae)

#Actual mark v prediction marks
results = pd.DataFrame({
    "Actual Marks": y_test.values,
    "Predicted Marks": y_pred
})
results["Error"] = abs(
    results["Actual Marks"] - results["Predicted Marks"]
)
print("\nActual vs Predicted Results:")
print(results.round(2))


plt.xlabel("Student")
plt.ylabel("Marks")
plt.title("Student Performance Comparison")

plt.show()


plt.scatter(df["study_hours"], df["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()
