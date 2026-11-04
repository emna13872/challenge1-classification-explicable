"""Recours contrefactuel : quel petit changement réaliste ferait passer un client de « refus » à « accord » ?

Règles (voir la phase B de la feuille de route) :
  1. seules les variables que le CLIENT peut changer sont touchées (liste déclarée) ; jamais une variable sensible ;
  2. chaque valeur essayée est une valeur OBSERVÉE chez de vrais clients, au plus au 95e centile (ou au-dessus du 5e
     pour une baisse) : un montant rare est observé, mais ce n'est pas un conseil raisonnable ;
  3. au plus `max_changes` variables à la fois, les plus petits changements possibles (distance en centiles :
     comparable d'une variable à l'autre, quelle que soit l'unité) ;
  4. chaque conseil est re-vérifié par le chemin complet de l'application (préparation puis modèle) ;
  5. si rien de réaliste ne marche, on le dit : « aucun conseil trouvé », avec le risque le plus bas atteint.
Le conseil décrit ce que fait le MODÈLE ; il ne garantit ni le comportement futur du client ni la décision de la banque.
"""
import itertools

import numpy as np

from . import explain as X
from .reasons import _fmt_value, _label
from .state import Session

# Variables qu'un client peut vraiment modifier, par jeu de données ("up" : on ne propose que d'augmenter).
# Taïwan : seulement les montants remboursés. Les factures, retards passés, limite, âge, sexe, études, situation
# maritale restent fixes (la limite est décidée par la banque ; le passé ne se réécrit pas ; une facture n'est
# qu'une corrélation pour le modèle : la proposer encouragerait à jouer avec lui).
ACTIONABLE = {"taiwan_credit": {f"PAY_AMT{k}": "up" for k in range(1, 7)}}

CAVEAT = ("Ce conseil décrit ce qui ferait changer la réponse du modèle. Il ne garantit ni le comportement futur "
          "du client ni la décision finale de la banque.")
SPARSITY_PENALTY = 0.02       # coût d'une variable de plus : à effet égal, on préfère changer moins de choses


def _numeric_col(sess: Session, var: str):
    cols = [j for j, m in enumerate(sess.meta) if m["orig"] == var]
    if len(cols) == 1 and sess.meta[cols[0]]["kind"] == "num":
        return cols[0]
    return None


def _candidates(col, v0: float, direction: str, n: int, cap: float = 0.95):
    """Valeurs observées entre la position actuelle et le 99e (ou 1er) centile : [(valeur, coût en centiles)]."""
    f0 = float(np.mean(col <= v0))
    out = {}
    if direction in ("up", "both") and f0 < cap:
        for q in np.linspace(f0, cap, n + 1)[1:]:
            v = float(np.quantile(col, q))
            if v > v0:
                out[v] = abs(float(np.mean(col <= v)) - f0)
    if direction in ("down", "both") and f0 > 1 - cap:
        for q in np.linspace(f0, 1 - cap, n + 1)[1:]:
            v = float(np.quantile(col, q))
            if v < v0:
                out[v] = abs(float(np.mean(col <= v)) - f0)
    return sorted(out.items())


def _nice(v: float, up: bool, digits: int) -> float:
    """Arrondit à `digits` chiffres significatifs, dans le sens du changement (le conseil reste au moins aussi fort)."""
    if v == 0:
        return 0.0
    mag = 10 ** (int(np.floor(np.log10(abs(v)))) - digits + 1)
    return float((np.ceil(v / mag) if up else np.floor(v / mag)) * mag)


def _predict(sess, est, M0, changes: dict[int, np.ndarray]):
    """Probabilités de défaut pour un lot de lignes : M0 dont certaines colonnes prennent les valeurs données."""
    n = len(next(iter(changes.values())))
    rows = np.tile(M0, (n, 1))
    for j, vals in changes.items():
        rows[:, j] = vals
    return est.predict_proba(sess.scaler.transform(rows))[:, 1]


def recourse(sess: Session, name: str, thr: float, test_index=None, record=None, actionable=None,
             max_changes: int = 2, n_options: int = 3, n_candidates: int = 30, max_percentile: float = 0.95) -> dict:
    if name not in sess.models:
        raise ValueError(f"Modèle « {name} » non entraîné.")
    if not 0 < thr < 1:
        raise ValueError("Le seuil doit être strictement entre 0 et 1.")
    if max_changes not in (1, 2):
        raise ValueError("max_changes doit valoir 1 ou 2.")
    rules = dict(actionable) if actionable else dict(ACTIONABLE.get(sess.name, {}))
    for v, d in rules.items():
        if d not in ("up", "down", "both"):
            raise ValueError(f"Sens inconnu pour « {v} » : {d}")
        if X.is_sensitive(v):
            raise ValueError(f"« {v} » est une variable sensible : on ne propose jamais de la changer.")
        if v not in sess.features:
            raise ValueError(f"« {v} » n'est pas une variable du modèle.")
    usable = {v: d for v, d in rules.items() if _numeric_col(sess, v) is not None}

    M, Z, raw, _ = X.client_matrices(sess, test_index, record)
    est = sess.models[name]["est"]
    M0 = M[0]
    p0 = float(est.predict_proba(Z)[0, 1])
    base = {"probability": p0, "threshold": float(thr), "actionable": usable, "max_changes": max_changes,
            "caveat": CAVEAT, "options": [], "lowest_probability_found": None}
    if not usable:
        return {**base, "status": "no_actionable"}
    if p0 < thr:
        return {**base, "status": "already_accepted"}

    cand = {}
    for v, d in usable.items():
        j = _numeric_col(sess, v)
        cand[v] = (j, _candidates(sess.M_train[:, j], float(M0[j]), d, n_candidates, max_percentile))
    cand = {v: c for v, c in cand.items() if c[1]}

    found, lowest = [], p0
    for v, (j, cs) in cand.items():                                              # un seul changement
        vals, costs = np.array([c[0] for c in cs]), np.array([c[1] for c in cs])
        p = _predict(sess, est, M0, {j: vals})
        lowest = min(lowest, float(p.min()))
        for k in np.where(p < thr)[0]:
            found.append((costs[k] + SPARSITY_PENALTY, [(v, float(vals[k]), float(costs[k]))], float(p[k])))
    if max_changes == 2:
        for (va, (ja, ca)), (vb, (jb, cb)) in itertools.combinations(cand.items(), 2):
            A, B = np.meshgrid(np.arange(len(ca)), np.arange(len(cb)), indexing="ij")
            A, B = A.ravel(), B.ravel()
            va_, vb_ = np.array([c[0] for c in ca])[A], np.array([c[0] for c in cb])[B]
            cost = np.array([c[1] for c in ca])[A] + np.array([c[1] for c in cb])[B]
            p = _predict(sess, est, M0, {ja: va_, jb: vb_})
            lowest = min(lowest, float(p.min()))
            for k in np.where(p < thr)[0]:
                found.append((cost[k] + 2 * SPARSITY_PENALTY,
                              [(va, float(va_[k]), float(np.array([c[1] for c in ca])[A[k]])),
                               (vb, float(vb_[k]), float(np.array([c[1] for c in cb])[B[k]]))], float(p[k])))

    single_ok = {(chg[0][0], chg[0][1]) for _, chg, _ in found if len(chg) == 1}
    found = [t for t in found if len(t[1]) == 1 or not any((v, val) in single_ok for v, val, _ in t[1])]
    found.sort(key=lambda t: t[0])
    options, seen, unverified = [], set(), 0
    for score, chg, p_model in found:
        key = tuple(sorted(v for v, _, _ in chg))
        if key in seen:
            continue
        verified = None
        for digits in (2, 3, None):                                              # arrondi humain, sinon valeur exacte
            vals = {v: (_nice(val, val > float(M0[_numeric_col(sess, v)]), digits) if digits else round(val, 2))
                    for v, val, _ in chg}
            raw2 = dict(raw)
            raw2.update(vals)                                                    # re-vérification par le chemin complet
            _, Z2, _, _ = X.client_matrices(sess, record=raw2)
            p2 = float(est.predict_proba(Z2)[0, 1])
            if p2 < thr:
                verified = (vals, p2)
                break
        if verified is None:
            unverified += 1
            continue
        vals, p2 = verified
        seen.add(key)
        changes = []
        for v, val, c in chg:
            j = _numeric_col(sess, v)
            new_val = vals[v]
            changes.append({"feature": v, "label": _label(sess, v), "from": float(M0[j]), "to": new_val,
                            "percentile_from": float(np.mean(sess.M_train[:, j] <= M0[j])),
                            "percentile_to": float(np.mean(sess.M_train[:, j] <= new_val))})
        text = " et ".join(f"« {c['label']} » passait de {_fmt_value(sess, c['feature'], c['from'])} "
                           f"à {_fmt_value(sess, c['feature'], c['to'])}" for c in changes)
        options.append({"changes": changes, "new_probability": p2, "score": float(score), "verified": True,
                        "text": f"Si {text}, le risque annoncé passerait de {p0:.1%} à {p2:.1%} "
                                f"(sous le seuil de {thr:.1%})."})
        if len(options) >= n_options:
            break
    status = "found" if options else "none_found"
    return {**base, "status": status, "options": options, "lowest_probability_found": float(lowest),
            "n_unverified": unverified}
