# Genomic Knowledge Graphs & Heterogeneous Deep Learning

This repository provides reproducible implementations of heterogeneous Graph Neural Networks (GNNs) and biological data harmonization pipelines designed for non-coding RNA interactome modeling and gene regulatory network discovery.

## Repository Overview

- **`models/hetero_gnn.py`**: A modular heterogeneous graph neural network encoder implementing `GATv2Conv` within PyTorch Geometric's `HeteroConv`, natively processing relational edge attributes (13 dimensions) and node-specific layer normalization.
- **`pipelines/gene_identifier_map.py`**: High-throughput entity disambiguation mapping STRING protein IDs (`ENSP`) to Ensembl Gene IDs (`ENSG`) via the `MyGeneInfo` API, with string-score filtering ($\ge 700$) and undirected edge deduplication.
- **`pipelines/build_hetero_data.py`**: Topology conversion utility partitioning monolithic graph representations into strongly-typed `HeteroData` graphs spanning 6 node classes (lncRNA, miRNA, TF, PCG, circRNA, snoRNA) and 18 interaction topologies.

## Dependencies

- Python $\ge$ 3.10
- PyTorch $\ge$ 2.0
- PyTorch Geometric (`torch_geometric`)
- pandas, numpy, scikit-learn, mygene

*Note: Raw interaction datasets (LncTarD, STRING DB) and internal experimental logs remain private due to academic collaboration agreements.*
