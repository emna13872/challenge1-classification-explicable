"""Validation croisée répétée pour COMPARER les modèles.

Règle d'or : ce module ne lit jamais le lot de test (sess.X_test_raw / sess.y_test).
On choisit ici, sur le lot d'apprentissage ; le lot de test ne sert qu'à annoncer la note finale.
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from . import models as MD
from .prep import ScaleFirst
from .state import Session


def _subsample(X, y, max_rows: int, seed: int):
    """Sous-échantillon stratifié (mêmes proportions de classes) pour garder des temps raisonnables."""
    if len(y) <= max_rows:
        return X, np.asarray(y)
    Xs, _, ys, _ = train_test_split(X, np.asarray(y), train_size=max_rows, stratify=np.asarray(y),
                                    random_state=seed)
    return Xs, ys


def _summary(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def cross_validate(sess: Session, specs: list[dict], n_splits: int = 5, n_repeats: int = 2, seed: int = 0,
                   balance: bool = False, max_rows: int = 20000, refit_prep: bool = True) -> dict:
    """Compare des modèles par validation croisée stratifiée répétée, sur le lot d'apprentissage seul.

    refit_prep=True  : à chaque pli, l'imputation, l'encodage et la mise à l'échelle sont REAJUSTÉS
                       sur les seules données d'apprentissage du pli (méthode correcte).
    refit_prep=False : on réutilise la préparation ajustée sur tout le lot d'apprentissage
                       (plus rapide, mais le pli de validation a « participé » aux statistiques : fuite légère).
                       Gardé uniquement pour mesurer l'effet de cette fuite.
    """
    if sess.X_train_raw is None:
        raise ValueError("Préparez d'abord les données (module 2).")
    X, y = _subsample(sess.X_train_raw, sess.y_train, max_rows, seed)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    Z_all = None
    if not refit_prep:
        Z_all = sess.scaler.transform(sess.imp.transform(X).astype(float))

    res = {s["name"]: {"auc": [], "pr_auc": [], "seconds": 0.0} for s in specs}
    for tr, va in cv.split(X, y):
        ytr, yva = y[tr], y[va]
        if refit_prep:
            imp = clone(sess.imp)                                   # copie NON ajustée, même configuration
            Mtr = imp.fit_transform(X.iloc[tr]).astype(float)
            Mva = imp.transform(X.iloc[va]).astype(float)
            sc = ScaleFirst(sess.prep_cfg.get("scaling", "standard"), sess.n_num).fit(Mtr)
            Ztr, Zva = sc.transform(Mtr), sc.transform(Mva)
        else:
            Ztr, Zva = Z_all[tr], Z_all[va]
        pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
        for s in specs:
            t0 = time.time()
            est = MD.make(s["name"], s.get("params"), balance, pos_w)
            est.fit(Ztr, ytr)
            p = est.predict_proba(Zva)[:, 1]
            res[s["name"]]["auc"].append(roc_auc_score(yva, p))
            res[s["name"]]["pr_auc"].append(average_precision_score(yva, p))
            res[s["name"]]["seconds"] += time.time() - t0

    out = {}
    for name, r in res.items():
        out[name] = {"label": MD.CATALOG[name]["label"], "auc": _summary(r["auc"]),
                     "pr_auc": _summary(r["pr_auc"]), "seconds": round(r["seconds"], 1)}
    best = max(out, key=lambda k: out[k]["auc"]["mean"])
    margin = out[best]["auc"]["std"]
    for name, r in out.items():
        # « à égalité » : à moins d'un écart-type du meilleur ; indicatif, pas un test statistique
        r["tied_with_best"] = bool(r["auc"]["mean"] >= out[best]["auc"]["mean"] - margin)
        r["best"] = name == best
    return {"models": out, "best": best,
            "settings": {"n_splits": n_splits, "n_repeats": n_repeats, "n_fits_per_model": n_splits * n_repeats,
                         "n_rows_used": int(len(y)), "balance": balance, "refit_prep": refit_prep},
            "test_set_used": False}
