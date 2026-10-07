"""
Genomic ID Harmonization and High-Confidence PPI Network Processing.
Maps STRING protein identifiers (ENSP) to Ensembl gene identifiers (ENSG)
using MyGeneInfo and filters interactions by confidence score threshold.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from mygene import MyGeneInfo


def map_ensp_to_ensg(ensp_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Query MyGeneInfo API in batches to resolve ENSP protein IDs to ENSG gene IDs.
    """
    mg = MyGeneInfo()
    query_results = mg.querymany(
        ensp_list,
        scopes="ensembl.protein",
        fields="ensembl.gene",
        species="human",
    )

    mapping = {}
    for entry in query_results:
        if entry.get("notfound", False):
            continue
        ensp = entry.get("query")
        field = entry.get("ensembl")

        if isinstance(field, dict):
            mapping[ensp] = field.get("gene")
        elif isinstance(field, list) and len(field) > 0:
            mapping[ensp] = field[0].get("gene")
        else:
            mapping[ensp] = None

    return mapping


def filter_and_collapse_ppi(
    df: pd.DataFrame,
    confidence_threshold: int = 700,
) -> pd.DataFrame:
    """
    Filters STRING interactions by confidence score and deduplicates undirected edges.
    """
    filtered = df[df["score"] >= confidence_threshold].copy()

    # Sort identifiers alphabetically to ensure undirected uniqueness
    gene_pairs = np.sort(filtered[["ensg1", "ensg2"]].values, axis=1)
    filtered[["ensg1", "ensg2"]] = gene_pairs

    return filtered.drop_duplicates(subset=["ensg1", "ensg2"])