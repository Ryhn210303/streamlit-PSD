import streamlit as st
import pandas as pd
##from sklearn.preprocessing import LabelEncoder, StandardScaler
##from sklearn.tree import DecisionTreeClassifier
##from sklearn.naive_bayes import GaussianNB
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Title of the app
st.title("Milk Quality Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    # Data Preprocessing
    st.write("### Data Preprocessing:")
    st.write("Checking for null values...")
    st.write(data.isnull().sum())

    # Encoding categorical features
    encoder = LabelEncoder()
    data['Taste'] = encoder.fit_transform(data['Taste'])
    data['Odor'] = encoder.fit_transform(data['Odor'])
    data['Grade'] = data['Grade'].map({'low': 0, 'medium': 1, 'high': 2})

    # Standardizing numerical features
    scaler = StandardScaler()
    data[['pH', 'Temprature', 'Colour']] = scaler.fit_transform(data[['pH', 'Temprature', 'Colour']])

    # Splitting data
    X = data.drop('Grade', axis=1)
    y = data['Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred_dt = dt_model.predict(X_test)
    y_pred_nb = nb_model.predict(X_test)

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)

    # Display Results
    st.write("### Model Evaluation:")
    st.write(f"**Decision Tree Accuracy:** {accuracy_dt:.2f}")
    st.write(f"**Naive Bayes Accuracy:** {accuracy_nb:.2f}")

    st.write("### Decision Tree Classification Report:")
    st.text(classification_report(y_test, y_pred_dt))

    st.write("### Naive Bayes Classification Report:")
    st.text(classification_report(y_test, y_pred_nb))

    # Visualization
    st.write("### Performance Comparison:")
    metrics = ['Accuracy']
    values_dt = [accuracy_dt]
    values_nb = [accuracy_nb]

    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.35
    index = np.arange(len(metrics))

    ax.bar(index, values_dt, bar_width, label='Decision Tree')
    ax.bar(index + bar_width, values_nb, bar_width, label='Naive Bayes')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    st.pyplot(fig)
