"""Recours contrefactuel : quel changement réaliste ferait passer un client de « refus » à « accord » ?

Deux modes.

MODE « AVENIR » (jeux avec un historique mensuel, comme Taïwan) : le passé ne se réécrit pas, ni les retards ni
les montants déjà remboursés. La seule question honnête est : « si le client refait une demande dans m mois, et que
d'ici là il paie à temps en remboursant au moins R par mois, quel serait son risque ? ». Pour le modèle, les m mois
nouveaux entrent dans la fenêtre de 6 mois et poussent les plus anciens dehors ; le reste de l'historique et les
variables fixes (limite, âge, sexe, études, situation maritale) ne bougent pas.
Hypothèses, toutes écrites dans la réponse : les factures des nouveaux mois restent au niveau du dernier mois connu ;
R couvre au moins le minimum exigé par la banque (inconnu dans les données), donc aucun retard ; si R couvre la
facture entière, le statut est « payé en totalité » (code -1), sinon « payé sans retard, solde reporté » (code 0).

MODE « MAINTENANT » (liste de variables modifiables déclarée, sans historique) : règles d'origine :
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
import pandas as pd

from . import explain as X
from .prep import row_to_matrices
from .reasons import _fmt_value, _label
from .state import Session

# Mode « maintenant » : variables qu'un client peut modifier tout de suite, par jeu de données.
# Taïwan n'en a AUCUNE : toutes ses variables sont soit fixes (limite décidée par la banque, âge, sexe…), soit
# du passé (retards, factures, montants remboursés). Taïwan passe donc par le mode « avenir » ci-dessous.
ACTIONABLE: dict[str, dict[str, str]] = {}

# Mode « avenir » : description de la fenêtre mensuelle (mois 1 = le plus récent).
TIME_WINDOWS = {
    "taiwan_credit": {"months": 6, "status": "PAY_{k}", "bill": "BILL_AMT{k}", "paid": "PAY_AMT{k}",
                      "code_full": -1, "code_partial": 0, "code_none": -2},
}
HORIZONS = (1, 3, 6)          # dans combien de mois le client pourrait refaire une demande

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


def _pct(p: float, thr: float) -> str:
    """Pourcentage à 1 décimale ; 2 décimales si l'arrondi ferait croire que p égale le seuil."""
    a, b = f"{p:.1%}", f"{thr:.1%}"
    return f"{p:.2%}" if a == b else a


def _predict(sess, est, M0, changes: dict[int, np.ndarray]):
    """Probabilités de défaut pour un lot de lignes : M0 dont certaines colonnes prennent les valeurs données."""
    n = len(next(iter(changes.values())))
    rows = np.tile(M0, (n, 1))
    for j, vals in changes.items():
        rows[:, j] = vals
    return est.predict_proba(sess.scaler.transform(rows))[:, 1]


def _check(sess: Session, name: str, thr: float):
    if name not in sess.models:
        raise ValueError(f"Modèle « {name} » non entraîné.")
    if not 0 < thr < 1:
        raise ValueError("Le seuil doit être strictement entre 0 et 1.")


def _money(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def _shifted(raw: dict, w: dict, m: int, payment: float, bill: float) -> dict:
    """La fenêtre de 6 mois vue dans m mois : m mois nouveaux (sans retard, `payment` remboursé), les anciens décalés."""
    n = w["months"]
    if bill <= 0:
        code = w["code_none"]                                                    # rien à payer ce mois-là
    else:
        code = w["code_full"] if payment >= bill else w["code_partial"]
    out = dict(raw)
    for key in ("status", "bill", "paid"):
        old = {k: raw.get(w[key].format(k=k)) for k in range(1, n + 1)}
        for k in range(n, m, -1):                                                # le mois k-m d'avant devient le mois k
            out[w[key].format(k=k)] = old[k - m]
        for k in range(1, min(m, n) + 1):
            out[w[key].format(k=k)] = {"status": code, "bill": bill, "paid": payment}[key]
    return out


def payment_candidates(ref: np.ndarray, bill: float, n: int = 40, max_percentile: float = 0.95) -> list[float]:
    """Montants mensuels à essayer : observés chez de vrais clients, arrondis, au plus au 95e centile et à la facture.

    Facture nulle : rien à payer, le seul scénario est 0. Facture positive : jamais 0 (ne rien payer = un retard).
    """
    if bill <= 0:
        return [0.0]
    cap = _nice(float(np.quantile(ref, max_percentile)), True, 2)
    out = {_nice(float(q), True, 2) for q in np.quantile(ref, np.linspace(0, max_percentile, n)) if q > 0}
    if bill <= cap:
        out.add(float(bill))                                                     # payer la facture en entier
    return sorted(c for c in out if 0 < c <= min(cap, bill))


def future_recourse(sess: Session, name: str, thr: float, test_index=None, record=None,
                    horizons=HORIZONS, n_candidates: int = 40, max_percentile: float = 0.95) -> dict:
    """Mode « avenir » : pour chaque délai m, le plus petit remboursement mensuel R qui ferait passer sous le seuil.

    R est cherché parmi des montants observés chez de vrais clients (au plus au 95e centile), arrondis, et jamais
    au-dessus de la facture : on ne rembourse pas plus que ce qu'on doit.
    """
    _check(sess, name, thr)
    w = TIME_WINDOWS[sess.name]
    n = w["months"]
    paid_cols = [w["paid"].format(k=k) for k in range(1, n + 1)]
    need = paid_cols + [w[key].format(k=k) for key in ("status", "bill") for k in range(1, n + 1)]
    est = sess.models[name]["est"]
    base = {"mode": "future", "threshold": float(thr), "caveat": CAVEAT, "options": [], "horizons": list(horizons),
            "lowest_probability_found": None, "n_unverified": 0,
            "actionable": {"paiements sans retard (mois à venir)": "up",
                           "montant remboursé chaque mois (mois à venir)": "up"}}
    if any(c not in sess.features for c in need):
        M, Z, raw, _ = X.client_matrices(sess, test_index, record)
        return {**base, "probability": float(est.predict_proba(Z)[0, 1]), "status": "no_actionable",
                "actionable": {}, "assumptions": []}
    M, Z, raw, _ = X.client_matrices(sess, test_index, record)
    p0 = float(est.predict_proba(Z)[0, 1])
    habit = float(np.mean([float(raw[c]) for c in paid_cols]))
    bill = max(0.0, float(raw[w["bill"].format(k=1)]))
    base.update({
        "probability": p0, "habit_monthly_payment": habit, "assumed_bill": bill,
        "assumptions": [
            f"les factures des prochains mois restent au niveau du dernier mois connu ({_money(bill)}) ;",
            "le montant proposé couvre au moins le minimum exigé par la banque (inconnu dans les données), "
            "donc aucun retard ;",
            "le reste de l'historique et les variables fixes (limite, âge, sexe, études, situation maritale) "
            "ne changent pas.",
        ]})
    if p0 < thr:
        return {**base, "status": "already_accepted"}

    ref = sess.X_train_raw[paid_cols[0]].to_numpy(float)                         # distribution des remboursements
    cands = payment_candidates(ref, bill, n_candidates, max_percentile)
    if not cands:
        return {**base, "status": "none_found", "lowest_probability_found": p0}
    rows, keys = [], []
    for m in horizons:
        for r in cands:
            rows.append(_shifted(raw, w, m, r, bill))
            keys.append((m, r))
    df = pd.DataFrame(rows)[sess.features]
    for c in sess.features:
        if pd.api.types.is_numeric_dtype(sess.X_train_raw[c]) and not pd.api.types.is_bool_dtype(sess.X_train_raw[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    _, Zc = row_to_matrices(sess, df)                                           # chemin complet : préparation + modèle
    pc = est.predict_proba(Zc)[:, 1]
    options = []
    for m in horizons:
        ok = [(r, float(pc[i])) for i, (mm, r) in enumerate(keys) if mm == m and pc[i] < thr]
        if not ok:
            continue
        r, p2 = min(ok)                                                          # le plus petit montant qui suffit
        months = "1 mois" if m == 1 else f"{m} mois"
        if bill <= 0:
            how, extra = "n'a aucune facture impayée (comme aujourd'hui)", ""
        elif r >= bill:
            how = f"paie chaque facture en totalité ({_money(r)} par mois)"
            extra = f" Aujourd'hui, il rembourse {_money(habit)} par mois en moyenne."
        else:
            how = f"paie sans aucun retard en remboursant au moins {_money(r)} par mois"
            extra = (f" C'est moins qu'aujourd'hui ({_money(habit)} en moyenne) : l'essentiel est d'éviter tout retard."
                     if r <= habit else f" Aujourd'hui, il rembourse {_money(habit)} par mois en moyenne.")
        pf, pt = float(np.mean(ref <= habit)), float(np.mean(ref <= r))
        options.append({
            "horizon_months": m, "monthly_payment": r, "pays_full_bill": bool(r >= bill), "new_probability": p2,
            "verified": True, "score": float(max(0.0, pt - pf)),
            "changes": [{"feature": "PAY_AMT (mois à venir)", "label": "Montant remboursé chaque mois",
                         "from": habit, "to": r, "percentile_from": pf, "percentile_to": pt}],
            "text": (f"Nouvelle demande dans {months} : le risque annoncé serait de {_pct(p2, thr)} au lieu de "
                     f"{_pct(p0, thr)} (seuil {thr:.1%}) si, d'ici là, le client {how}.{extra}"),
        })
    status = "found" if options else "none_found"
    return {**base, "status": status, "options": options, "lowest_probability_found": float(min(pc.min(), p0))}


def recourse(sess: Session, name: str, thr: float, test_index=None, record=None, actionable=None,
             max_changes: int = 2, n_options: int = 3, n_candidates: int = 30, max_percentile: float = 0.95) -> dict:
    _check(sess, name, thr)
    if actionable is None and sess.name in TIME_WINDOWS:
        return future_recourse(sess, name, thr, test_index, record, max_percentile=max_percentile)
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
    base = {"mode": "now", "probability": p0, "threshold": float(thr), "actionable": usable, "max_changes": max_changes,
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
                        "text": f"Si {text}, le risque annoncé passerait de {_pct(p0, thr)} à {_pct(p2, thr)} "
                                f"(sous le seuil de {thr:.1%})."})
        if len(options) >= n_options:
            break
    status = "found" if options else "none_found"
    return {**base, "status": status, "options": options, "lowest_probability_found": float(lowest),
            "n_unverified": unverified}
