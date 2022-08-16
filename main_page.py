import streamlit as st

st.markdown("# Pulmonary Arterial Hypertension diagnostic system")
st.image("images/Machine_learning.jpg")
st.markdown("## Introduction")
st.sidebar.markdown("# Introduction")
st.text("")
st.markdown("A system that helps detect Pulmonary Arterial Hypertension (PAH) in patient's DNA microarray with the power of machine learning algorithms.")
st.text("")
st.markdown("### Types of PAH:")
st.markdown(
    """
    - Health control: Healthy patients.
    - HPAH: (Heritable Pulmonary Arterial Hypertension) type of PAH which occurs due to mutations in PAH predisposing genes or in a familial context.
    - IPAH: (Idiopathic Pulmonary Arterial Hypertension) type of PAH which occurs with no apparent cause.
    - UMC: (Unaffected Mutation Carriers) patients who carry genetic mutation but do not show any symptom of PAH.
    """)
st.text("")
st.markdown("### Supported functionalities:")
st.markdown("""
    - Diagnose your own data.
    - View top 10 genes identified by different feature selection algorithms.
    - View the distributions of the 10 most important genes according to SIRRFE algorithm.
""")


st.text("")
st.markdown("### How it works")
st.markdown("""
        Patient's DNA microarray is first transformed using logarithm base 2 (binary logarithm). Afterwards, the obtained expressions is scaled with pretrained mean and standard variation. Finally, we feed the data to our pretrained **Wide and Deep** model to yield the final result.
    """)
