import streamlit as st
from diagnose import end2end
import numpy as np

st.markdown("# Pulmonary Arterial Hypertension diagnostic system")
st.markdown("## Diagnose")
st.sidebar.markdown("# Diagnose")
st.text("")
st.markdown("Detect PAH with our customized **Wide and Deep** machine learning model")
file = st.file_uploader("Choose your file")
button = st.button("Detect")
if file is not None:
    ext = file.name.split(".")[-1]
    if ext == "xlsx":
        if button:
            probability, prediction, inf_time = end2end(file)
            if prediction == 0:
                result = "HPAH - Heritable Pulmonary Arterial Hypertension"
            elif prediction == 1:
                result = "Healthy Control"
            elif prediction == 2:
                result = "IPAH - Idiopathic Pulmonary Arterial Hypertension"
            else:
                result = "UMC - Unaffected Mutation Carriers"

            with st.form("Prediction result"):
                st.text("Patient is diagnosed with: {}".format(result))
                st.text("Probability of HPAH:                         {}".format(probability[0][0]))
                st.text("Probability of Healthy Control:              {}".format(probability[0][1]))
                st.text("Probability of IPAH:                         {}".format(probability[0][2]))
                st.text("Probability of UMC:                          {}".format(probability[0][3]))
                st.text("Inference time:                              {}s".format(np.round(inf_time, 4)))
                st.form_submit_button("Back")
    else: 
        st.warning("You need to upload a csv or excel file!")
else:
    if button:
        st.warning("Please upload your file!")