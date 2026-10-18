"""Module 2 – préparation : imputation, encodage, normalisation, découpage."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .data import binarize
from .state import Session


class ScaleFirst:
    """Normalise uniquement les n premières colonnes (les numériques)."""

    def __init__(self, kind: str, n: int):
        self.n = n
        self.kind = kind
        self.sk = {"standard": StandardScaler(), "minmax": MinMaxScaler()}.get(kind)

    def fit(self, M):
        if self.sk is not None and self.n:
            self.sk.fit(M[:, : self.n])
        return self

    def transform(self, M):
        M = np.asarray(M, dtype=float)
        if self.sk is None or not self.n:
            return M.copy()
        out = M.copy()
        out[:, : self.n] = self.sk.transform(M[:, : self.n])
        return out


def build(sess: Session, cfg: dict):
    df, target, positive = sess.df, sess.target, sess.positive
    feats = [f for f in cfg["features"] if f in df.columns and f != target]
    if not feats:
        raise ValueError("Sélectionnez au moins une variable explicative.")
    X = df[feats].copy()
    for c in X.columns:                      # booléens -> catégories
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(str)
    y = binarize(df[target], positive)
    keep = df[target].notna()
    if cfg.get("num_impute") == "drop" or cfg.get("cat_impute") == "drop":
        keep &= X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    if y.nunique() < 2:
        raise ValueError("La cible ne contient qu'une seule classe après filtrage.")

    test_size = float(cfg.get("test_size", 0.25))
    strat = y if cfg.get("stratify", True) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=int(cfg.get("random_state", 42)), stratify=strat)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    ni = cfg.get("num_impute", "median")
    num_strategy = {"mean": "mean", "median": "median", "zero": "constant", "drop": "median"}.get(ni, "median")
    num_imp = SimpleImputer(strategy=num_strategy, fill_value=0 if num_strategy == "constant" else None)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_imp, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    imp = ColumnTransformer(transformers, verbose_feature_names_out=False)

    M_train = imp.fit_transform(Xtr).astype(float)
    M_test = imp.transform(Xte).astype(float)

    names, meta = list(num_cols), [{"orig": c, "kind": "num", "category": None} for c in num_cols]
    if cat_cols:
        cats = imp.named_transformers_["cat"].named_steps["ohe"].categories_
        for c, cs in zip(cat_cols, cats):
            for v in cs:
                names.append(f"{c}_{v}")
                meta.append({"orig": c, "kind": "cat", "category": str(v)})
    assert len(names) == M_train.shape[1]

    scaler = ScaleFirst(cfg.get("scaling", "standard"), len(num_cols)).fit(M_train)

    sess.features = feats
    sess.prep_cfg = cfg
    sess.imp, sess.scaler, sess.n_num = imp, scaler, len(num_cols)
    sess.names, sess.meta = names, meta
    sess.X_train_raw, sess.X_test_raw = Xtr, Xte
    sess.y_train, sess.y_test = ytr, yte
    sess.M_train, sess.M_test = M_train, M_test
    sess.Z_train, sess.Z_test = scaler.transform(M_train), scaler.transform(M_test)
    sess.models, sess.cache = {}, {}
    return {
        "n_train": len(Xtr), "n_test": len(Xte), "n_features_in": len(feats),
        "n_features_encoded": len(names), "numeric": num_cols, "categorical": cat_cols,
        "train_pos_rate": float(ytr.mean()), "test_pos_rate": float(yte.mean()),
        "dropped_rows": int((~keep).sum()), "encoded_names": names,
    }


def row_to_matrices(sess: Session, rows: pd.DataFrame):
    """Transforme des lignes brutes en (espace affichage, espace modèle)."""
    rows = rows.copy()
    for c in rows.columns:
        if pd.api.types.is_bool_dtype(rows[c]):
            rows[c] = rows[c].astype(str)
    M = sess.imp.transform(rows[sess.features]).astype(float)
    return M, sess.scaler.transform(M)


def form_schema(sess: Session):
    """Description des champs pour le formulaire d'individu fictif."""
    out = []
    X = sess.X_train_raw
    for c in sess.features:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out.append({"name": c, "kind": "num", "min": float(s.min()), "max": float(s.max()),
                        "default": float(s.median()), "integer": bool((s.dropna() % 1 == 0).all())})
        else:
            vc = s.astype(str).value_counts()
            out.append({"name": c, "kind": "cat", "choices": list(vc.index[:50]),
                        "default": str(vc.index[0])})
    return out
