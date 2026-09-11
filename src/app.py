import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import train
from model_eval import get_predictions
from de import test_loader


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
    hidden_channels = st.number_input(
        "Number hidden channels",
        min_value=1,
        max_value=100,
        value=32,
        step=1,
    )
    lr = st.number_input(
        "Learning rate",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%0.3f",
    )
    weight_decay = st.number_input(
        "Weight decay",
        min_value=0.0,
        max_value=0.01,
        value=0.01,
        step=0.00001,
        format="%0.5f",
    )
    num_layers = st.number_input(
        "Number of layers",
        min_value=1,
        max_value=12,
        value=2,
        step=1,
    )
    if st.button("Train", use_container_width=True):
        st.session_state.should_train = True

# run training with default vals
if st.session_state.get("should_train", False):
    with st.spinner("Training..."):
        st.session_state.model = train(
            num_epochs,
            hidden_channels=hidden_channels,
            lr=lr,
            weight_decay=weight_decay,
            num_layers=num_layers,
        )
    model = st.session_state.model

    if st.session_state.get("should_train", False):
        y_true, y_pred, mse = get_predictions(test_loader, model)
        st.session_state.y_true = y_true
        st.session_state.y_pred = y_pred
        st.session_state.mse = mse

        if st.session_state.mse < st.session_state.get("best_mse", float("inf")):
            st.session_state.best_mse = st.session_state.mse

    st.session_state.should_train = False  # reset after training

# plot
if "y_true" in st.session_state:

    y_true = st.session_state.y_true
    y_pred = st.session_state.y_pred

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

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

    #     st.divider()
    # if "mse" in st.session_state:
    st.metric("Current MSE", value=f"{st.session_state.mse:.4f}")
    st.metric("Best MSE",    value=f"{st.session_state.best_mse:.4f}")
    # else:
    #     st.caption("Run training to see metrics")

    st.session_state.should_train = False  # reset after training

else:
    st.info("Press Train to see results :)")