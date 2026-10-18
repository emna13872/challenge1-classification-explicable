"""Module 3 – catalogue de modèles, hyperparamètres et optimisation."""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

CATALOG = {
    "logreg": {
        "label": "Régression logistique", "kind": "linear",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100},
        },
        "grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
    "tree": {
        "label": "Arbre de décision", "kind": "tree",
        "params": {
            "max_depth": {"type": "int", "default": 5, "min": 1, "max": 30, "step": 1},
            "min_samples_leaf": {"type": "int", "default": 20, "min": 1, "max": 200, "step": 1},
        },
        "grid": {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [5, 20, 50]},
    },
    "knn": {
        "label": "k plus proches voisins", "kind": "knn",
        "params": {
            "n_neighbors": {"type": "int", "default": 25, "min": 1, "max": 200, "step": 1},
        },
        "grid": {"n_neighbors": [5, 15, 25, 51, 101]},
    },
    "rf": {
        "label": "Random Forest", "kind": "tree",
        "params": {
            "n_estimators": {"type": "int", "default": 200, "min": 50, "max": 1000, "step": 50},
            "max_depth": {"type": "int", "default": 10, "min": 2, "max": 40, "step": 1},
            "min_samples_leaf": {"type": "int", "default": 5, "min": 1, "max": 100, "step": 1},
        },
        "grid": {"max_depth": [6, 10, 16], "min_samples_leaf": [3, 10]},
    },
    "xgb": {
        "label": "XGBoost", "kind": "tree",
        "params": {
            "n_estimators": {"type": "int", "default": 300, "min": 50, "max": 1500, "step": 50},
            "max_depth": {"type": "int", "default": 4, "min": 1, "max": 12, "step": 1},
            "learning_rate": {"type": "float", "default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005},
        },
        "grid": {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]},
    },
}


def _coerce(name, params):
    spec = CATALOG[name]["params"]
    out = {}
    for k, v in (params or {}).items():
        if k in spec and v is not None:
            out[k] = int(round(float(v))) if spec[k]["type"] == "int" else float(v)
    return out


def make(name, params=None, balance=False, pos_weight=1.0):
    p = {**{k: v["default"] for k, v in CATALOG[name]["params"].items()}, **_coerce(name, params)}
    cw = "balanced" if balance else None
    if name == "logreg":
        return LogisticRegression(class_weight=cw, solver="lbfgs", **p)
    if name == "tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=0, **p)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=1, **p)
    if name == "rf":
        return RandomForestClassifier(class_weight=cw, random_state=0, n_jobs=-1, **p)
    if name == "xgb":
        return XGBClassifier(random_state=0, n_jobs=2, eval_metric="logloss", tree_method="hist",
                             scale_pos_weight=pos_weight if balance else 1.0, **p)
    raise ValueError(f"Modèle inconnu : {name}")


def train_one(name, params, tune, balance, Z, y, cv=3, max_tune_rows=8000):
    """Entraîne un modèle ; si `tune`, recherche en grille (AUC, validation croisée)."""
    t0 = time.time()
    pos_w = float((y == 0).sum() / max(1, (y == 1).sum()))
    info = {"tuned": bool(tune)}
    if tune:
        idx = np.arange(len(y))
        if len(idx) > max_tune_rows:
            idx = np.random.RandomState(0).choice(idx, max_tune_rows, replace=False)
        gs = GridSearchCV(make(name, params, balance, pos_w), CATALOG[name]["grid"],
                          scoring="roc_auc", cv=cv, n_jobs=1, refit=False)
        gs.fit(Z[idx], np.asarray(y)[idx])
        best = {**_coerce(name, params), **gs.best_params_}
        info["cv_auc"] = float(gs.best_score_)
        params = best
    est = make(name, params, balance, pos_w)
    est.fit(Z, y)
    used = {k: getattr(est, k) for k in CATALOG[name]["params"] if hasattr(est, k)}
    info.update({"params": used, "train_seconds": round(time.time() - t0, 2)})
    return est, info
