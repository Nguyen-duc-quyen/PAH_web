from soupsieve import select
import streamlit as st
import pandas as pd

st.markdown("# Pulmonary Arterial Hypertension diagnostic system")
st.markdown("## Gene Analysis")
st.sidebar.markdown("# Gene Analysis")

option = st.sidebar.selectbox(
    "Choose feature ranking method",
    ('ANOVA', 'SIRRFE', 'Machine Learning')
)

if option == "ANOVA":
    data_path = "checkpoints/Anova_top_10_genes.xlsx"
elif option == "SIRRFE":
    data_path = "checkpoints/Sirrfe_top_10_genes.xlsx"
else:
    data_path = "checkpoints/svm_top_10_genes.xlsx"

dataframe = pd.read_excel(data_path, header=0)
dataframe = dataframe[['gene', 'Accession']]
st.write("Top 10 genes by: ", option)
st.table(dataframe)