"""Seuil de décision choisi par le COÛT des erreurs (et non « 50 % par habitude »).

Deux erreurs possibles, qui ne coûtent pas la même chose :
  - cas manqué (FN) : on accorde le crédit à quelqu'un qui fait défaut -> coût cost_fn ;
  - fausse alerte (FP) : on refuse un client qui aurait remboursé   -> coût cost_fp.
Avec des probabilités FIABLES (calibrées), refuser est rentable dès que  p * cost_fn > (1 - p) * cost_fp,
soit  p > cost_fp / (cost_fp + cost_fn) : c'est le seuil théorique. On le vérifie sur les données.
Tout est calculé sur des probabilités hors-pli du lot d'apprentissage ; le lot de test n'est jamais lu.
"""
import numpy as np

from . import calibration as CAL
from . import models as MD
from .state import Session


def theoretical_threshold(cost_fn: float, cost_fp: float) -> float:
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("Les coûts doivent être strictement positifs.")
    return float(cost_fp / (cost_fp + cost_fn))


def cost_at(y, p, thr: float, cost_fn: float, cost_fp: float) -> dict:
    """Coût moyen par dossier si l'on refuse tous les dossiers dont la probabilité de défaut est >= thr."""
    y = np.asarray(y)
    refuse = np.asarray(p) >= thr
    fn = int(((~refuse) & (y == 1)).sum())
    fp = int((refuse & (y == 0)).sum())
    tp = int((refuse & (y == 1)).sum())
    n = len(y)
    return {"threshold": float(thr), "cost": (fn * cost_fn + fp * cost_fp) / n, "refuse_rate": float(refuse.mean()),
            "recall": tp / max(1, int((y == 1).sum())), "precision": tp / max(1, int(refuse.sum())),
            "fn": fn, "fp": fp}


def cost_curve(y, p, cost_fn: float, cost_fp: float) -> tuple[list[dict], dict]:
    """Coût pour chaque seuil de 0,01 à 0,99, et la ligne du meilleur seuil."""
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    curve = [cost_at(y, p, t, cost_fn, cost_fp) for t in grid]
    return curve, min(curve, key=lambda r: r["cost"])


def decision_report(sess: Session, spec: dict, cost_fn: float, cost_fp: float, n_splits: int = 5, seed: int = 0,
                    balance: bool = False, max_rows: int = 20000) -> dict:
    t_theo = theoretical_threshold(cost_fn, cost_fp)
    y, probs = CAL.oof_by_method(sess, spec, n_splits, seed, balance, max_rows)
    method = CAL.pick_best(y, probs)                 # on décide sur les probabilités les plus fiables
    p = probs[method]
    curve, best = cost_curve(y, p, cost_fn, cost_fp)
    rate = float(np.mean(y))
    return {
        "model": spec["name"], "label": MD.CATALOG[spec["name"]]["label"], "calibration_used": method,
        "costs": {"cost_fn": cost_fn, "cost_fp": cost_fp, "ratio": cost_fn / cost_fp},
        "base_rate": rate,
        "theoretical_threshold": t_theo,
        "best": best,                                              # seuil qui minimise le coût sur les données
        "at_theoretical": cost_at(y, p, t_theo, cost_fn, cost_fp),
        "at_half": cost_at(y, p, 0.5, cost_fn, cost_fp),           # le seuil « par habitude »
        "accept_all": rate * cost_fn,                              # on ne refuse personne
        "refuse_all": (1 - rate) * cost_fp,                        # on refuse tout le monde
        "curve": curve,
        "settings": {"n_splits": n_splits, "n_rows_used": int(len(y)), "balance": balance},
        "test_set_used": False,
    }
