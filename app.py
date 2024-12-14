import streamlit as st
import pickle
import numpy as np

# Load the trained model and symptom list
with open("disease_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    symptom_list = pickle.load(f)

# Streamlit UI
st.title("Personalized Healthcare System")

# Multi-select for symptoms
st.write("### Select your symptoms")
selected_symptoms = st.multiselect("Choose from the list below:", symptom_list)

# Button to predict
if st.button("Predict Diseases"):
    if selected_symptoms:
        # Prepare input vector
        input_vector = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            input_vector[symptom_list.index(symptom)] = 1

        # Predict disease
        predicted_disease = model.predict([input_vector])[0]
        st.write(f"### Predicted Disease: {predicted_disease}")

        # Show all potential diseases with probabilities
        probabilities = model.predict_proba([input_vector])[0]
        disease_probabilities = sorted(
            zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True
        )
        st.write("### Possible Diseases with Probabilities:")
        for disease, probability in disease_probabilities:
            st.write(f"{disease}: {probability:.2f}")
    else:
        st.write("Please select at least one symptom.")
