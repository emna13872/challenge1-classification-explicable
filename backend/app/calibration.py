"""Calibration : quand le modèle annonce 30 %, environ 30 % des cas sont-ils vraiment positifs ?

On mesure sur des probabilités HORS-PLI du lot d'apprentissage (jamais sur le lot de test) puis on compare
trois versions : brute, corrigée par sigmoïde (méthode de Platt) et corrigée par régression isotonique.
Pour juger honnêtement une correction, elle est elle-même ajustée sur une partie des données et jugée sur l'autre.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import models as MD
from . import validation as V
from .state import Session

EPS = 1e-6
MIN_GAIN = 0.0005          # en dessous, la correction ne vaut pas la peine : on garde les probabilités brutes


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def brier(y, p) -> float:
    """Erreur moyenne au carré entre la probabilité annoncée et la réalité (0 ou 1). Plus petit = mieux."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(y, p, n_bins: int = 10) -> list[dict]:
    """Groupes de clients de même effectif, triés par probabilité annoncée."""
    order = np.argsort(p)
    out = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            out.append({"n": int(len(idx)), "mean_pred": float(np.mean(np.asarray(p)[idx])),
                        "frac_pos": float(np.mean(np.asarray(y)[idx]))})
    return out


def ece(bins: list[dict]) -> float:
    """Expected Calibration Error : écart moyen, pondéré par l'effectif, entre annoncé et réel."""
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["mean_pred"] - b["frac_pos"]) for b in bins))


def _calibrate_cv(y, p, method: str, n_splits: int, seed: int) -> np.ndarray:
    """Ajuste la correction sur (n_splits-1) parts et l'applique à la part restante, à tour de rôle."""
    out = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1).split(p.reshape(-1, 1), y):
        if method == "sigmoid":
            cal = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p[tr]), y[tr])
            out[va] = cal.predict_proba(_logit(p[va]))[:, 1]
        else:
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[tr], y[tr])
            out[va] = cal.predict(p[va])
    return out


def fit_calibrator(y, p, method: str):
    """Ajuste la correction sur toutes les prédictions hors-pli ; renvoie une fonction proba brute -> proba corrigée."""
    y, p = np.asarray(y), np.asarray(p)
    if method == "sigmoid":
        cal = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p), y)
        return lambda q: cal.predict_proba(_logit(np.asarray(q)))[:, 1]
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return lambda q: cal.predict(np.asarray(q))
    return lambda q: np.asarray(q, dtype=float)


def _metrics(y, p) -> dict:
    bins = reliability(y, p)
    return {"brier": brier(y, p), "ece": ece(bins), "mean_pred": float(np.mean(p)), "bins": bins}


def oof_by_method(sess: Session, spec: dict, n_splits: int = 5, seed: int = 0, balance: bool = False,
                  max_rows: int = 20000) -> tuple[np.ndarray, dict]:
    """Probabilités hors-pli sous les trois versions : brute, sigmoïde, isotonique (lot d'apprentissage seul)."""
    y, p = V.oof_predictions(sess, spec, n_splits=n_splits, seed=seed, balance=balance, max_rows=max_rows)
    return y, {"raw": p, "sigmoid": _calibrate_cv(y, p, "sigmoid", n_splits, seed),
               "isotonic": _calibrate_cv(y, p, "isotonic", n_splits, seed)}


def pick_best(y, probs: dict) -> str:
    """La version au Brier le plus bas ; la brute si le gain est négligeable."""
    scores = {k: brier(y, v) for k, v in probs.items()}
    best = min(scores, key=scores.get)
    return "raw" if scores["raw"] - scores[best] < MIN_GAIN else best


def calibration_report(sess: Session, spec: dict, n_splits: int = 5, seed: int = 0, balance: bool = False,
                       max_rows: int = 20000) -> dict:
    y, probs = oof_by_method(sess, spec, n_splits, seed, balance, max_rows)
    methods = {k: _metrics(y, v) for k, v in probs.items()}
    best = pick_best(y, probs)
    rate = float(np.mean(y))
    return {"model": spec["name"], "label": MD.CATALOG[spec["name"]]["label"], "methods": methods,
            "recommended": best, "base_rate": rate, "base_brier": rate * (1 - rate),
            "settings": {"n_splits": n_splits, "n_rows_used": int(len(y)), "balance": balance},
            "test_set_used": False}
