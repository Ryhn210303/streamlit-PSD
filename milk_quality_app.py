import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Example usage
st.title("Example")
data = {"Taste": ["good", "bad", "good"], "Odor": ["strong", "weak", "strong"]}
df = pd.DataFrame(data)

# Encode categorical columns
encoder = LabelEncoder()
df['Taste'] = encoder.fit_transform(df['Taste'])
df['Odor'] = encoder.fit_transform(df['Odor'])

st.write(df)
