import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd

st.markdown("# Pulmonary Arterial Hypertension diagnostic system")
st.markdown("## Gene Distribution")
st.sidebar.markdown("# Gene Distribution")
st.text("")
st.write("The distributions of the top 10 most important genes founded by the SIRRFE algorithm")
st.markdown("*Suppose that the genes follow normal (Gaussian) distribution*")
st.text("")

option = st.sidebar.selectbox(
    "Choose feature ranking method",
    ('ANOVA', 'SIRRFE')
)

if option == "ANOVA":
    mean_path = "checkpoints/anova_mean_genes_group.csv"
    std_path = "checkpoints/anova_std_genes_group.csv"
else :
    mean_path = "checkpoints/mean_genes_group.csv"
    std_path = "checkpoints/std_genes_group.csv"


# load data
mean_genes = pd.read_csv(mean_path, header=0)
std_genes = pd.read_csv(std_path, header=0)
genes = mean_genes.iloc[:, 0].values

# Create 10 plots for 10 genes
figure, axis = plt.subplots(5, 2, figsize=(20, 20))
# Draw each gene distribution
for i in range(1,11):
    row = int((i-1)%5)
    column = int((i-1)/5)
    #print("{}, {}".format(row, column))
    # 4 group
    mu_1 = mean_genes.iloc[-i, 1]
    mu_2 = mean_genes.iloc[-i, 2]
    mu_3 = mean_genes.iloc[-i, 3]
    mu_4 = mean_genes.iloc[-i, 4]
    
    variance_1 = std_genes.iloc[-i, 1]
    variance_2 = std_genes.iloc[-i, 2]
    variance_3 = std_genes.iloc[-i, 3]
    variance_4 = std_genes.iloc[-i, 4]
    
    sigma_1 = math.sqrt(variance_1)
    sigma_2 = math.sqrt(variance_2)
    sigma_3 = math.sqrt(variance_3)
    sigma_4 = math.sqrt(variance_4)

    x_1 = np.linspace(mu_1 - 3*sigma_1, mu_1 + 3*sigma_1, 100)
    x_2 = np.linspace(mu_2 - 3*sigma_2, mu_2 + 3*sigma_2, 100)
    x_3 = np.linspace(mu_3 - 3*sigma_3, mu_3 + 3*sigma_3, 100)
    x_4 = np.linspace(mu_4 - 3*sigma_4, mu_4 + 3*sigma_4, 100)

    axis[row, column].plot(x_1, stats.norm.pdf(x_1, mu_1, sigma_1), color= 'green', label='Healthy')     #Healthy control
    axis[row, column].plot(x_2, stats.norm.pdf(x_2, mu_2, sigma_2), color= 'red', label='HPAH')       #HPAH
    axis[row, column].plot(x_3, stats.norm.pdf(x_3, mu_3, sigma_3), color= 'blue', label='UMC')      #UMC
    axis[row, column].plot(x_4, stats.norm.pdf(x_4, mu_4, sigma_4), color= 'yellow', label='IPAH')    #IPAH
    axis[row, column].set_title("Gene: {}".format(genes[-i]))

    if row == 4 and column == 1:
        axis[row, column].legend(loc='lower right')


#figure.savefig('images/anova_gene_distributions.png', bbox_inches='tight')
st.pyplot(figure)
