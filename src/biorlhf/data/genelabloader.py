"""
GeneLab fGSEA/GSVA data loading for BioGRPO.

Loads pathway enrichment results from the GeneLab_benchmark project's
processed fGSEA and GSVA files. Provides consensus pathway directions
across missions for use as verifiable ground truth.
"""

import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
import json

import pandas as pd

# ── Paths (configurable via env vars for HPC) ─────────────────────────────
GENELAB_BASE = Path(os.environ.get(
    "GENELAB_BASE",
    "/Users/jak4013/Dropbox/Bioinformatics/Claude/GeneLab_benchmark",
))
FGSEA_DIR = GENELAB_BASE / "processed" / "fgsea"
GSVA_DIR = GENELAB_BASE / "processed" / "pathway_scores"
TASKS_DIR = GENELAB_BASE / "tasks"
EVAL_DIR = GENELAB_BASE / "evaluation"

# ── Tissue → available missions (from actual files) ───────────────────────
TISSUE_MISSIONS: Dict[str, List[str]] = {
    "liver": ["MHU-2", "RR-1", "RR-3", "RR-6", "RR-8", "RR-9"],
    "gastrocnemius": ["RR-1", "RR-9"],
    "kidney": ["RR-1", "RR-3", "RR-7"],
    "thymus": ["MHU-2", "RR-6", "RR-9"],
    "skin": ["MHU-2_dorsal", "MHU-2_femoral", "RR-6"],
    "eye": ["RR-1", "RR-3"],
}

# Tissue → LOMO task ID
TISSUE_TASK_MAP: Dict[str, str] = {
    "liver": "A1",
    "gastrocnemius": "A2",
    "kidney": "A3",
    "thymus": "A4",
    "skin": "A5",
    "eye": "A6",
}

DBS = ["hallmark", "kegg", "reactome", "mitocarta"]


@dataclass
class PathwayResult:
    """Single pathway enrichment result from fGSEA."""
    pathway: str
    nes: float
    padj: float
    direction: str          # "UP", "DOWN", or "NS"
    tissue: str
    mission: str
    db: str
    leading_edge: List[str] = field(default_factory=list)


# ── Loading functions ──────────────────────────────────────────────────────

def load_fgsea(tissue: str, mission: str, db: str = "hallmark") -> pd.DataFrame:
    """Load a single fGSEA result CSV.

    Returns DataFrame with columns:
        pathway, pval, padj, log2err, ES, NES, size, db,
        leadingEdge_str, tissue, mission, glds
    """
    path = FGSEA_DIR / tissue / f"{mission}_fgsea_{db}.csv"
    if not path.exists():
        raise FileNotFoundError(f"fGSEA file not found: {path}")
    return pd.read_csv(path)


def load_all_fgsea(tissue: str, db: str = "hallmark") -> pd.DataFrame:
    """Load all fGSEA results for a tissue across all available missions."""
    dfs = []
    for mission in TISSUE_MISSIONS.get(tissue, []):
        path = FGSEA_DIR / tissue / f"{mission}_fgsea_{db}.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_pathway_directions(
    tissue: str,
    db: str = "hallmark",
    padj_threshold: float = 0.05,
) -> Dict[str, Dict[str, str]]:
    """Return pathway directions per mission.

    Returns:
        {mission: {pathway: "UP"/"DOWN"/"NS"}}
        Only pathways with padj < threshold get UP/DOWN; rest are NS.
    """
    df = load_all_fgsea(tissue, db)
    if df.empty:
        return {}

    result: Dict[str, Dict[str, str]] = {}
    for mission, mdf in df.groupby("mission"):
        directions: Dict[str, str] = {}
        for _, row in mdf.iterrows():
            if pd.notna(row["padj"]) and row["padj"] < padj_threshold:
                directions[row["pathway"]] = "UP" if row["NES"] > 0 else "DOWN"
            else:
                directions[row["pathway"]] = "NS"
        result[str(mission)] = directions
    return result


def get_consensus_directions(
    tissue: str,
    db: str = "hallmark",
    min_missions: int = 2,
    padj_threshold: float = 0.05,
) -> Dict[str, Dict]:
    """Return pathways with consensus direction across missions.

    Only includes pathways where >= min_missions agree on direction
    and the majority direction has more votes than the opposite.

    Returns:
        {pathway: {
            direction: "UP"/"DOWN",
            n_agree: int,
            n_disagree: int,
            n_ns: int,
            missions_agree: List[str],
            missions_disagree: List[str],
        }}
    """
    all_dirs = get_pathway_directions(tissue, db, padj_threshold)
    if not all_dirs:
        return {}

    # Collect per-pathway votes
    pathway_votes: Dict[str, Dict[str, List[str]]] = {}
    for mission, pmap in all_dirs.items():
        for pathway, direction in pmap.items():
            if pathway not in pathway_votes:
                pathway_votes[pathway] = {"UP": [], "DOWN": [], "NS": []}
            pathway_votes[pathway][direction].append(mission)

    consensus: Dict[str, Dict] = {}
    for pathway, votes in pathway_votes.items():
        n_up = len(votes["UP"])
        n_down = len(votes["DOWN"])
        n_ns = len(votes["NS"])

        if n_up >= min_missions and n_up > n_down:
            consensus[pathway] = {
                "direction": "UP",
                "n_agree": n_up,
                "n_disagree": n_down,
                "n_ns": n_ns,
                "missions_agree": votes["UP"],
                "missions_disagree": votes["DOWN"],
            }
        elif n_down >= min_missions and n_down > n_up:
            consensus[pathway] = {
                "direction": "DOWN",
                "n_agree": n_down,
                "n_disagree": n_up,
                "n_ns": n_ns,
                "missions_agree": votes["DOWN"],
                "missions_disagree": votes["UP"],
            }
    return consensus


def get_disagreeing_pathways(
    tissue: str,
    db: str = "hallmark",
    padj_threshold: float = 0.05,
) -> Dict[str, Dict]:
    """Return pathways where missions disagree on direction.

    These are ideal for uncertainty questions — the model should
    express uncertainty about direction.

    Returns:
        {pathway: {
            missions_up: List[str],
            missions_down: List[str],
            missions_ns: List[str],
        }}
    """
    all_dirs = get_pathway_directions(tissue, db, padj_threshold)
    if not all_dirs:
        return {}

    pathway_votes: Dict[str, Dict[str, List[str]]] = {}
    for mission, pmap in all_dirs.items():
        for pathway, direction in pmap.items():
            if pathway not in pathway_votes:
                pathway_votes[pathway] = {"UP": [], "DOWN": [], "NS": []}
            pathway_votes[pathway][direction].append(mission)

    disagreeing: Dict[str, Dict] = {}
    for pathway, votes in pathway_votes.items():
        if votes["UP"] and votes["DOWN"]:
            disagreeing[pathway] = {
                "missions_up": votes["UP"],
                "missions_down": votes["DOWN"],
                "missions_ns": votes["NS"],
            }
    return disagreeing


def load_gsva_scores(
    tissue: str,
    mission: str,
    db: str = "hallmark",
) -> pd.DataFrame:
    """Load GSVA pathway scores (samples × pathways)."""
    path = GSVA_DIR / tissue / f"{mission}_gsva_{db}.csv"
    if not path.exists():
        raise FileNotFoundError(f"GSVA file not found: {path}")
    return pd.read_csv(path, index_col=0)


def load_lomo_splits(tissue: str) -> List[Dict]:
    """Load LOMO fold definitions from task_info.json."""
    task_id = TISSUE_TASK_MAP.get(tissue)
    if not task_id:
        return []
    task_dir = TASKS_DIR / f"{task_id}_{tissue}_lomo"
    info_path = task_dir / "task_info.json"
    if not info_path.exists():
        return []
    with open(info_path) as f:
        info = json.load(f)
    return info.get("folds", [])


def load_nes_conservation(db: str = "hallmark") -> Dict:
    """Load NES conservation analysis (cross-mission correlation data)."""
    path = EVAL_DIR / f"NES_conservation_{db}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_all_pathways(tissue: str, db: str = "hallmark") -> List[str]:
    """Get sorted list of all pathway names for a tissue/db combo."""
    df = load_all_fgsea(tissue, db)
    if df.empty:
        return []
    return sorted(df["pathway"].unique().tolist())


def get_pathway_nes_matrix(
    tissue: str,
    db: str = "hallmark",
) -> pd.DataFrame:
    """Return a mission × pathway NES matrix for a tissue.

    Useful for visualizing pathway behavior across missions.
    """
    df = load_all_fgsea(tissue, db)
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index="mission", columns="pathway", values="NES", aggfunc="first",
    )
