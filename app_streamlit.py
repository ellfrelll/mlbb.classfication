import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_mlbb.joblib")

st.title("MLBB Classification")
kill = st.slider("jumlah Kill",0,20)
assist = st.slider("jumlah Assist",0,20)
death = st.slider("jumlah Death",0,20)
turret = st.slider("jumlah Turret",0,20)

if st.button("predict"):
	data_baru = pd.DataFrame([[kill,assist,death,turret]],columns =["kill","assist","death","turret"])
	st.success(f"Hasil Prediction : {model.predict(data_baru)[0]}")
	