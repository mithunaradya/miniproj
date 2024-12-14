from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import pickle

# Load the dataset
processed_file = "Processed_Disease_Prediction_Dataset.csv"
data = pd.read_csv(processed_file)

# Features (symptoms) and target (prognosis)
X = data.iloc[:, :-1]  # All columns except 'prognosis'
y = data["prognosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
with open("disease_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the feature names (symptoms)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and symptom list saved as 'disease_prediction_model.pkl' and 'symptom_list.pkl'")
