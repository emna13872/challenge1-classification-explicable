"""Motifs de refus (reason codes) : pourquoi ce client a-t-il ce score ?

À partir des contributions SHAP du modèle (combien chaque colonne pousse le score par rapport à un client moyen) :
  1. on regroupe les colonnes qui viennent de la même variable (un encodage one-hot en produit plusieurs) ;
  2. on écarte les variables sensibles : elles ne sont JAMAIS citées comme motif (elles restent dans l'audit d'équité) ;
  3. on garde les `top` variables qui poussent le plus vers le risque ;
  4. on écrit chaque motif avec la vraie valeur du client et une valeur de comparaison tirée des données d'apprentissage.
Tout est calculé : aucune phrase n'est inventée, ce qui permettra plus tard à un agent de les citer sans les déformer.
"""
import math

import numpy as np

from . import explain as X
from .state import Session


def _friendly_taiwan() -> dict:
    lab = {"LIMIT_BAL": "Limite de crédit accordée", "SEX": "Sexe", "EDUCATION": "Niveau d'études",
           "MARRIAGE": "Situation maritale", "AGE": "Âge"}
    when = {1: "du mois dernier", 2: "d'il y a 2 mois", 3: "d'il y a 3 mois", 4: "d'il y a 4 mois",
            5: "d'il y a 5 mois", 6: "d'il y a 6 mois"}
    for k, w in when.items():
        lab[f"PAY_{k}"] = f"Statut de remboursement {w}"
        lab[f"BILL_AMT{k}"] = f"Montant de la facture {w}"
        lab[f"PAY_AMT{k}"] = f"Montant remboursé {w}"
    return lab


LABELS = {"taiwan_credit": _friendly_taiwan()}


def _label(sess: Session, col: str) -> str:
    return LABELS.get(sess.name, {}).get(col, col)


def _fmt_value(sess: Session, orig: str, v: float) -> str:
    """Rend un code lisible quand on le connaît (Taïwan : PAY_k = retard de remboursement), sinon le nombre brut."""
    if sess.name == "taiwan_credit" and orig.startswith("PAY_") and not orig.startswith("PAY_AMT"):
        k = int(round(v))
        if k >= 1:
            return f"retard de {k} mois"
        if k == -1:
            return "payé à temps"
        return f"code {k} (non documenté)"
    return X._fmt(v)


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _groups(sess: Session) -> dict:
    g = {}
    for j, m in enumerate(sess.meta):
        g.setdefault(m["orig"], []).append(j)
    return g


def _describe(sess: Session, orig: str, cols: list[int], m_row) -> dict:
    """Valeur du client et valeur de comparaison, pour UNE variable d'origine."""
    meta = sess.meta[cols[0]]
    label = _label(sess, orig)
    if meta["kind"] == "num":
        j = cols[0]
        value, typical = float(m_row[j]), float(np.median(sess.M_train[:, j]))
        text = (f"{label} : {_fmt_value(sess, orig, value)} "
                f"(valeur médiane des clients : {_fmt_value(sess, orig, typical)})")
        return {"value": value, "typical": typical, "kind": "num", "text_core": text}
    active = [j for j in cols if m_row[j] >= 0.5]
    if active:
        j = active[0]
        cat, share = sess.meta[j]["category"], float(np.mean(sess.M_train[:, j]))
        text = f"{label} : « {cat} » (observé chez {share:.0%} des clients)"
        return {"value": cat, "typical": share, "kind": "cat", "text_core": text}
    return {"value": None, "typical": None, "kind": "cat", "text_core": f"{label} : autre modalité"}


def decision_reasons(sess: Session, name: str, thr: float, test_index=None, record=None, top: int = 3) -> dict:
    if name not in sess.models:
        raise ValueError(f"Modèle « {name} » non entraîné.")
    if not 0 < thr < 1:
        raise ValueError("Le seuil doit être strictement entre 0 et 1.")
    M, Z, _raw, true_label = X.client_matrices(sess, test_index, record)
    m_row = M[0]
    est = sess.models[name]["est"]
    p = float(est.predict_proba(Z)[0, 1])
    sv, base = X.shap_values(sess, name, Z)
    sv = sv[0]
    space = X.shap_space(name)

    items, excluded = [], []
    for orig, cols in _groups(sess).items():
        contrib = float(sv[cols].sum())
        if X.is_sensitive(orig):
            excluded.append(orig)
            continue
        d = _describe(sess, orig, cols, m_row)
        items.append({"feature": orig, "label": _label(sess, orig), "contribution": contrib, **d})

    against = sorted((i for i in items if i["contribution"] > 0), key=lambda i: -i["contribution"])[:top]
    favour = sorted((i for i in items if i["contribution"] < 0), key=lambda i: i["contribution"])[:top]
    total_up = sum(i["contribution"] for i in items if i["contribution"] > 0) or 1.0
    for i in against + favour:
        i["share_of_push"] = abs(i["contribution"]) / (total_up if i["contribution"] > 0 else
                                                       (sum(-j["contribution"] for j in items if j["contribution"] < 0) or 1.0))
        i["text"] = i.pop("text_core")
    for i in items:
        i.pop("text_core", None)

    out_model = base + float(sv.sum())
    target = p if space == "probabilité" else _logit(p)
    return {
        "model": name, "probability": p, "threshold": float(thr),
        "decision": "refus" if p >= thr else "accord", "true_label": true_label,
        "space": space, "base_value": base,
        "reasons_against": against, "reasons_for": favour,
        "excluded_sensitive": excluded,
        "additivity_gap": abs(out_model - target),          # doit être ~0 : les contributions s'additionnent bien
        "n_variables": len(items),
    }
