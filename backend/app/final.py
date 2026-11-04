"""Évaluation finale : le SEUL endroit où le lot de test est lu pour juger la chaîne complète.

Tout ce qui précède (choix du modèle, correction de calibration, seuil) est décidé sur le lot d'apprentissage seul :
  1. probabilités hors-pli du modèle choisi ;
  2. correction de calibration ajustée sur ces probabilités (si elle améliore le Brier) ;
  3. seuil qui minimise le coût sur ces probabilités corrigées.
Ensuite le modèle est entraîné sur tout l'apprentissage et on ouvre le lot de test UNE fois, sans rien ré-ajuster.
Chaque ouverture est comptée : rouvrir le test pour changer d'avis le transforme en lot de choix (biais de sélection).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import calibration as CAL
from . import decision as DEC
from . import models as MD
from .state import Session


def _bootstrap_ci(y, p, thr, cost_fn, cost_fp, reps: int = 400, seed: int = 0) -> list[float]:
    """Intervalle à 95 % du coût moyen sur le lot de test (rééchantillonnage des dossiers)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = [DEC.cost_at(y[i], p[i], thr, cost_fn, cost_fp)["cost"] for i in (rng.integers(0, n, n) for _ in range(reps))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def final_evaluation(sess: Session, spec: dict, cost_fn: float, cost_fp: float, n_splits: int = 5, seed: int = 0,
                     balance: bool = False, max_rows: int = 20000, force: bool = False) -> dict:
    if sess.Z_test is None or sess.y_test is None:
        raise ValueError("Préparez d'abord les données (module 2).")
    if sess.final is not None and not force:
        return {**sess.final, "already_opened": True}

    DEC.theoretical_threshold(cost_fn, cost_fp)                       # valide les coûts
    # 1-3 : tout se décide sur le lot d'apprentissage
    y, probs = CAL.oof_by_method(sess, spec, n_splits, seed, balance, max_rows)
    method = CAL.pick_best(y, probs)
    p_oof = probs[method]
    _, best_oof = DEC.cost_curve(y, p_oof, cost_fn, cost_fp)
    thr = best_oof["threshold"]
    calibrate = CAL.fit_calibrator(y, probs["raw"], method)

    # modèle final : tout l'apprentissage
    ytr = np.asarray(sess.y_train)
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    est = MD.make(spec["name"], spec.get("params"), balance, pos_w)
    est.fit(sess.Z_train, ytr)

    # ouverture du lot de test (comptée)
    sess.test_openings += 1
    yt = np.asarray(sess.y_test)
    p_raw = est.predict_proba(sess.Z_test)[:, 1]
    p_cal = calibrate(p_raw)
    rate = float(yt.mean())
    at = DEC.cost_at(yt, p_cal, thr, cost_fn, cost_fp)
    ci = _bootstrap_ci(yt, p_cal, thr, cost_fn, cost_fp)
    res = {
        "model": spec["name"], "label": MD.CATALOG[spec["name"]]["label"], "calibration_used": method,
        "costs": {"cost_fn": cost_fn, "cost_fp": cost_fp, "ratio": cost_fn / cost_fp},
        "threshold": thr, "theoretical_threshold": DEC.theoretical_threshold(cost_fn, cost_fp),
        "expected": {**best_oof, "brier": CAL.brier(y, p_oof), "n": int(len(y))},     # ce que la validation croisée promettait
        "test": {
            "n": int(len(yt)), "base_rate": rate,
            "auc": float(roc_auc_score(yt, p_raw)), "pr_auc": float(average_precision_score(yt, p_raw)),
            "brier_raw": CAL.brier(yt, p_raw), "brier_calibrated": CAL.brier(yt, p_cal),
            "ece_raw": CAL.ece(CAL.reliability(yt, p_raw)), "ece_calibrated": CAL.ece(CAL.reliability(yt, p_cal)),
            "bins_raw": CAL.reliability(yt, p_raw), "bins_calibrated": CAL.reliability(yt, p_cal),
            "at_threshold": at, "cost_ci95": ci,
            "at_half": DEC.cost_at(yt, p_cal, 0.5, cost_fn, cost_fp),
            "accept_all": rate * cost_fn, "refuse_all": (1 - rate) * cost_fp,
            "expected_inside_ci": bool(ci[0] <= best_oof["cost"] <= ci[1]),
        },
        "settings": {"n_splits": n_splits, "n_rows_used": int(len(y)), "balance": balance},
        "test_set_used": True, "openings": sess.test_openings, "already_opened": False,
        "warning": None if sess.test_openings == 1 else
                   f"Le lot de test a été ouvert {sess.test_openings} fois : cette note n'est plus tout à fait honnête.",
    }
    sess.final = res
    return res
