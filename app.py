import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.train import train
from src.eval import get_predictions
from src.de import test_loader


# page setup
st.set_page_config(page_title="Molecular GNN experimentation", layout="wide")
st.title("Molecular GNN experimentation")

# sidebar
with st.sidebar:
    st.header("Settings")
    num_epochs = st.number_input(
        "Number of epochs",
        min_value=1,
        max_value=1000,
        value=100,
        step=1,
    )
    run = st.button("Train", use_container_width=True)

# run training with default vals
if "model" not in st.session_state or run:
    with st.spinner("Training..."):
        st.session_state.model = train(num_epochs)

model = st.session_state.model

# TODO: type of gnn

# stats


# plot
fig, ax = plt.subplots()
fig.patch.set_facecolor("white")

y_true, y_pred = get_predictions(test_loader, model)

ax.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')

ax.set_xlabel('Actual Y Values', color="black")
ax.set_ylabel('Predicted Y Values', color="black")
ax.set_facecolor("white")
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

st.pyplot(fig)