"""Suivi de dérive (phase F) : les nouveaux clients ressemblent-ils encore à ceux de l'apprentissage ?

Un modèle apprend le monde d'hier. Si les clients changent (crise, nouvelle offre, autre région), ses probabilités
deviennent fausses sans qu'aucune erreur ne s'affiche. Le PSI (Population Stability Index, indice de stabilité
d'une population) mesure l'écart entre deux distributions d'une même variable :
    PSI = somme sur les tranches de (part_actuelle − part_référence) × ln(part_actuelle / part_référence)
Tranches : les déciles de la référence (variables numériques) ou les modalités (variables catégorielles).
Seuils d'usage dans le crédit (convention, pas une loi) : < 0,10 stable ; 0,10 à 0,25 à surveiller ; > 0,25 dérive.
On calcule le PSI de chaque variable ET celui du score (la probabilité annoncée par le modèle) : c'est le score
que la banque utilise pour décider, donc l'alerte la plus importante.
"""
import numpy as np
import pandas as pd

from .prep import row_to_matrices
from .state import Session

EPS = 1e-4                                   # évite ln(0) quand une tranche est vide d'un côté
STABLE, WATCH = 0.10, 0.25


def level(psi: float) -> str:
    return "stable" if psi < STABLE else ("à surveiller" if psi < WATCH else "dérive")


def _shares_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10):
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # variable presque constante : on compare les valeurs elles-mêmes
        return _shares_categorical(pd.Series(ref).astype(str), pd.Series(cur).astype(str))
    edges[0], edges[-1] = -np.inf, np.inf    # les valeurs hors de la plage de référence comptent aux extrémités
    r = np.histogram(ref, edges)[0] / max(1, len(ref))
    c = np.histogram(cur, edges)[0] / max(1, len(cur))
    return r, c


def _shares_categorical(ref: pd.Series, cur: pd.Series):
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return r, c


def psi_from_shares(r: np.ndarray, c: np.ndarray) -> float:
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi(ref, cur, categorical: bool = False, bins: int = 10) -> float:
    if categorical:
        return psi_from_shares(*_shares_categorical(pd.Series(ref), pd.Series(cur)))
    return psi_from_shares(*_shares_numeric(np.asarray(ref, float), np.asarray(cur, float), bins))


def shifted(sess: Session, feature: str, factor: float = 1.0, offset: float = 0.0) -> pd.DataFrame:
    """Un lot de test volontairement décalé (pour vérifier que l'alerte se déclenche) : x → x × facteur + décalage."""
    if feature not in sess.features:
        raise ValueError(f"Variable « {feature} » absente de la préparation.")
    X = sess.X_test_raw.copy()
    if not pd.api.types.is_numeric_dtype(X[feature]):
        raise ValueError(f"« {feature} » n'est pas numérique : on ne peut pas la multiplier.")
    X[feature] = X[feature] * float(factor) + float(offset)
    return X


def drift_report(sess: Session, model: str | None = None, current: pd.DataFrame | None = None,
                 threshold: float = 0.5, label: str = "lot de test") -> dict:
    """Compare l'apprentissage (référence) à un lot « actuel » : chaque variable, puis le score du modèle."""
    if sess.X_train_raw is None:
        raise ValueError("Préparez d'abord les données (module 2).")
    ref = sess.X_train_raw
    cur = sess.X_test_raw if current is None else current
    rows = []
    for f in sess.features:
        cat = not pd.api.types.is_numeric_dtype(ref[f]) or pd.api.types.is_bool_dtype(ref[f])
        p = psi(ref[f], cur[f], categorical=cat)
        rows.append({"feature": f, "psi": p, "level": level(p), "kind": "catégorielle" if cat else "numérique",
                     "ref_median": None if cat else float(np.nanmedian(ref[f])),
                     "cur_median": None if cat else float(np.nanmedian(cur[f]))})
    rows.sort(key=lambda r: -r["psi"])
    out = {"label": label, "n_reference": int(len(ref)), "n_current": int(len(cur)), "features": rows,
           "n_drift": sum(r["level"] == "dérive" for r in rows),
           "n_watch": sum(r["level"] == "à surveiller" for r in rows),
           "thresholds": {"stable": STABLE, "watch": WATCH}}
    if model and model in sess.models:
        est = sess.models[model]["est"]
        p_ref = est.predict_proba(sess.Z_train)[:, 1]
        Z_cur = sess.Z_test if current is None else row_to_matrices(sess, cur[sess.features])[1]
        p_cur = est.predict_proba(Z_cur)[:, 1]
        s = psi(p_ref, p_cur)
        out["score"] = {"psi": s, "level": level(s), "mean_reference": float(p_ref.mean()),
                        "mean_current": float(p_cur.mean()), "threshold": float(threshold),
                        "refusal_rate_reference": float((p_ref >= threshold).mean()),
                        "refusal_rate_current": float((p_cur >= threshold).mean())}
    worst = rows[0] if rows else None
    out["alert"] = bool(out["n_drift"] or (out.get("score", {}).get("level") == "dérive"))
    out["summary"] = ("Aucune dérive : le lot ressemble à l'apprentissage." if not out["alert"] else
                      f"Dérive détectée : {out['n_drift']} variable(s) au-dessus de {WATCH}"
                      + (f", la plus forte sur « {worst['feature']} » (PSI {worst['psi']:.2f})" if worst else "")
                      + (f" ; score du modèle : PSI {out['score']['psi']:.2f}" if "score" in out else "") + ".")
    return out
