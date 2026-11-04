"""Audit du recours : sur TOUS les clients refusés, combien reçoivent un conseil, de quelle taille, et est-ce équitable ?

Un conseil trouvé pour un client ne prouve rien pour les autres. Cet audit répond à trois questions :
  1. Quelle part des clients refusés reçoit un conseil réaliste, et quelle part n'en reçoit aucun ?
  2. Les conseils demandent-ils des efforts raisonnables (taille du changement) ?
  3. L'accès au conseil est-il le même pour tous les groupes (sexe, tranche d'âge) ?
     Les variables sensibles ne servent jamais de motif ni de conseil, mais elles peuvent QUAND MÊME être corrélées
     avec les autres : seule une mesure par groupe le révèle.
"""
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import explain as X
from . import recourse as RC
from .state import Session

MIN_GROUP = 30               # en dessous, un pourcentage par groupe ne veut rien dire
GAP_ALERT = 0.10             # écart d'accès au conseil entre deux groupes à partir duquel on alerte


def wilson(k: int, n: int, z: float = 1.96):
    """Intervalle de confiance à 95 % d'une proportion k/n (méthode de Wilson : reste dans [0, 1], même si n est petit).

    Sert quand on audite un ÉCHANTILLON de clients : « 58 % » sur 200 clients veut dire « entre 51 % et 65 % environ ».
    """
    if n == 0:
        return None
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    lo, hi = max(0.0, mid - half), min(1.0, mid + half)
    return [0.0 if k == 0 else float(lo), 1.0 if k == n else float(hi)]       # bornes exactes aux extrêmes


def _groups(sess: Session, col: str, idx: np.ndarray):
    """Groupes d'une variable sensible pour les clients de `idx` : valeurs si peu nombreuses, sinon quartiles."""
    s = sess.X_test_raw[col].iloc[idx]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 6:
        edges = np.unique(np.quantile(sess.X_train_raw[col].dropna(), [0, .25, .5, .75, 1]))
        if len(edges) < 3:
            return None
        if np.allclose(edges, np.round(edges)):                                   # entiers (âge) : « 21–28 », « 29–34 »…
            e = [int(round(x)) for x in edges]
            names = [f"{e[0]}–{e[1]}"] + [f"{e[i] + 1}–{e[i + 1]}" for i in range(1, len(e) - 1)]
        else:
            names = [f"{edges[0]:g} à {edges[1]:g}"] + [f"plus de {edges[i]:g} à {edges[i + 1]:g}"
                                                       for i in range(1, len(edges) - 1)]
        lab = pd.cut(s, bins=edges, include_lowest=True, labels=names)
        return lab.astype(str).to_numpy()
    return s.astype(str).to_numpy()


def audit(sess: Session, name: str, thr: float, max_clients: int | None = 400, seed: int = 0,
          actionable=None, max_changes: int = 2) -> dict:
    if name not in sess.models:
        raise ValueError(f"Modèle « {name} » non entraîné.")
    if not 0 < thr < 1:
        raise ValueError("Le seuil doit être strictement entre 0 et 1.")
    t0 = time.time()
    p = sess.models[name]["est"].predict_proba(sess.Z_test)[:, 1]
    refused = np.where(p >= thr)[0]
    out = {"threshold": float(thr), "n_test": int(len(p)), "n_refused": int(len(refused)),
           "share_refused": float(len(refused) / max(1, len(p))), "caveat": RC.CAVEAT}
    if len(refused) == 0:
        return {**out, "status": "no_refused", "n_audited": 0}
    sampled = max_clients is not None and len(refused) > max_clients
    idx = np.sort(np.random.default_rng(seed).choice(refused, max_clients, replace=False)) if sampled else refused
    res = [RC.recourse(sess, name, thr, test_index=int(i), actionable=actionable, max_changes=max_changes)
           for i in idx]
    n = len(res)
    status = Counter(r["status"] for r in res)
    if status.get("no_actionable") == n:
        return {**out, "status": "no_actionable", "n_audited": n, "sampled": bool(sampled)}
    found = [r for r in res if r["status"] == "found"]
    none = [r for r in res if r["status"] == "none_found"]
    out.update({"status": "ok", "n_audited": n, "sampled": bool(sampled),
                "share_found": len(found) / n, "share_none": len(none) / n,
                "share_found_ci95": wilson(len(found), n),
                "n_found": len(found), "n_none": len(none),
                "n_unverified": int(sum(r.get("n_unverified", 0) for r in res))})

    best = [r["options"][0] for r in found]
    jumps = [c["percentile_to"] - c["percentile_from"] for o in best for c in o["changes"]]
    ratios = [c["to"] / c["from"] for o in best for c in o["changes"] if c["from"] > 0]
    out["effort"] = {
        "share_one_change": float(np.mean([len(o["changes"]) == 1 for o in best])) if best else None,
        "median_percentile_jump": float(np.median(jumps)) if jumps else None,
        "p90_percentile_jump": float(np.quantile(jumps, 0.9)) if jumps else None,
        "median_ratio": float(np.median(ratios)) if ratios else None,
        "share_from_zero": float(np.mean([c["from"] == 0 for o in best for c in o["changes"]])) if best else None,
        "variables_used": dict(Counter(c["feature"] for o in best for c in o["changes"]).most_common()),
    }
    if best and "horizon_months" in best[0]:                                     # mode « avenir »
        h = Counter(o["horizon_months"] for o in best)                           # le délai le plus court qui suffit
        out["effort"]["horizon_months"] = {str(k): h[k] for k in sorted(h)}
        out["effort"]["share_pays_full_bill"] = float(np.mean([o["pays_full_bill"] for o in best]))
    out["none_found"] = {
        "median_probability": float(np.median([r["probability"] for r in none])) if none else None,
        "median_lowest_probability": float(np.median([r["lowest_probability_found"] for r in none])) if none else None,
        "median_probability_when_found": float(np.median([r["probability"] for r in found])) if found else None,
    }
    if none and "assumed_bill" in none[0]:                                       # mode « avenir »
        out["none_found"]["share_no_bill"] = float(np.mean([r["assumed_bill"] <= 0 for r in none]))
        out["none_found"]["median_bill"] = float(np.median([r["assumed_bill"] for r in none]))
        out["none_found"]["median_bill_when_found"] = float(np.median([r["assumed_bill"] for r in found])) if found else None
    out["mode"] = res[0].get("mode")

    # équité de l'accès au conseil, par groupe sensible
    fair = []
    for col in sess.features:
        if not X.is_sensitive(col):
            continue
        g = _groups(sess, col, idx)
        if g is None:
            continue
        got = np.array([r["status"] == "found" for r in res])
        rows = [{"group": k, "n": int((g == k).sum()), "share_found": float(got[g == k].mean()),
                 "ci95": wilson(int(got[g == k].sum()), int((g == k).sum()))}
                for k in sorted(set(g)) if (g == k).sum() >= MIN_GROUP]
        if len(rows) >= 2:
            gap = max(r["share_found"] for r in rows) - min(r["share_found"] for r in rows)
            fair.append({"variable": col, "groups": rows, "gap": float(gap), "alert": bool(gap >= GAP_ALERT)})
    out["fairness"] = fair
    out["seconds"] = round(time.time() - t0, 1)
    return out
