# molecular-gnn
GNNs for molecular datasets.

GNN regression for the ESOL dataset with a streamlit app that lets you play around with hyperparams to see how they affect model performance (mse)

Inspired by "A gentle introduction to gaph neural networks" (see below)

# running with docker
`docker build -t molecular-gnn .` then `docker run --rm -p 8501:8501 molecular-gnn` and open http://localhost:8501. (or whatever port you choose)
ESOL is downloaded on container start, so the first page load takes a few seconds. To run locally instead: `streamlit run src/app.py`.

# sources
Sanchez-Lengeling, et al., "A Gentle Introduction to Graph Neural Networks", Distill, 2021.
https://pubs.acs.org/doi/10.1021/ci034243x - ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure, John S. Delaney
