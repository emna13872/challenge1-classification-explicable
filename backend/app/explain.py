"""Module 5 – Explainable AI : importances, SHAP, LIME, phrase de synthèse, audit d'équité."""
import re

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .models import CATALOG
from .prep import row_to_matrices
from .state import Session


# ---------------------------------------------------------------- utilitaires
def _fmt(v):
    v = float(v)
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def col_label(sess: Session, j: int, m_row) -> str:
    m = sess.meta[j]
    if m["kind"] == "num":
        return f"{m['orig']} = {_fmt(m_row[j])}"
    return f"{m['orig']} = {m['category']}" if m_row[j] >= 0.5 else f"{m['orig']} ≠ {m['category']}"


def proba(sess, name, Z):
    return sess.models[name]["est"].predict_proba(Z)[:, 1]


def _pos(sv):
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def _base(ev):
    ev = np.asarray(ev).ravel()
    return float(ev[1] if ev.size > 1 else ev[0])


# ---------------------------------------------------------------- SHAP
def _explainer(sess, name):
    key = (name, "explainer")
    if key in sess.cache:
        return sess.cache[key]
    est = sess.models[name]["est"]
    kind = CATALOG[name]["kind"]
    rs = np.random.RandomState(0)
    n = len(sess.Z_train)
    bg = sess.Z_train[rs.choice(n, min(200, n), replace=False)]
    if kind == "tree":
        ex = shap.TreeExplainer(est)
    elif kind == "linear":
        ex = shap.LinearExplainer(est, bg)
    else:
        ex = shap.KernelExplainer(lambda z: est.predict_proba(z)[:, 1], shap.kmeans(bg, 10))
    sess.cache[key] = ex
    return ex


def shap_space(name):
    return "probabilité" if name in ("tree", "rf", "knn") else "log-odds"


def shap_values(sess, name, Z):
    ex = _explainer(sess, name)
    if CATALOG[name]["kind"] == "knn":
        sv = ex.shap_values(Z, nsamples=120, silent=True)
    else:
        sv = ex.shap_values(Z)
    return _pos(sv), _base(ex.expected_value)


def _color(col):
    lo, hi = np.percentile(col, 5), np.percentile(col, 95)
    if hi == lo:
        lo, hi = col.min(), col.max()
    if hi == lo:
        return np.full(len(col), 0.5)
    return np.clip((col - lo) / (hi - lo), 0, 1)


def shap_global(sess, name, top=15, n_dep=6):
    key = (name, "shap_global")
    if key in sess.cache:
        return sess.cache[key]
    n = len(sess.Z_test)
    cap = 60 if CATALOG[name]["kind"] == "knn" else 300
    idx = np.random.RandomState(0).choice(n, min(n, cap), replace=False)
    sv, base = shap_values(sess, name, sess.Z_test[idx])
    M = sess.M_test[idx]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    total = mean_abs.sum() or 1.0
    bee = [{
        "feature": sess.names[j], "mean_abs": float(mean_abs[j]), "share": float(mean_abs[j] / total),
        "shap": sv[:, j].tolist(), "color": _color(M[:, j]).tolist(), "value": M[:, j].tolist(),
    } for j in order[:top]]
    dep = [{
        "feature": sess.names[j], "x": M[:, j].tolist(), "shap": sv[:, j].tolist(),
        "kind": sess.meta[j]["kind"],
    } for j in order[:n_dep]]
    share_by_orig = {}                       # part de l'impact SHAP par variable d'origine (toutes les colonnes)
    for j, m in enumerate(sess.meta):
        share_by_orig[m["orig"]] = share_by_orig.get(m["orig"], 0.0) + float(mean_abs[j] / total)
    res = {"space": shap_space(name), "base_value": base, "n_samples": int(len(idx)),
           "importance": [{"feature": sess.names[j], "mean_abs": float(mean_abs[j])} for j in order[:25]],
           "share_by_orig": share_by_orig,
           "beeswarm": bee, "dependence": dep}
    sess.cache[key] = res
    sess.cache[(name, "shap_raw")] = (idx, sv)
    return res


# ---------------------------------------------------------------- importances globales
def native_importance(sess, name, top=20):
    est = sess.models[name]["est"]
    kind = CATALOG[name]["kind"]
    if kind == "linear":
        v = est.coef_[0]
        label = "Coefficients (variables normalisées)" if sess.prep_cfg.get("scaling", "standard") != "none" \
            else "Coefficients (variables non normalisées : échelles non comparables)"
        signed = True
    elif kind == "tree":
        v = est.feature_importances_
        label = "Importance par impureté (gain)" if name == "xgb" else "Importance par impureté (MDI)"
        signed = False
    else:
        return {"available": False, "label": "Non disponible pour k-NN (modèle sans paramètres interprétables)",
                "items": []}
    order = np.argsort(-np.abs(v))[:top]
    return {"available": True, "label": label, "signed": signed,
            "items": [{"feature": sess.names[j], "value": float(v[j])} for j in order]}


def permutation(sess, name, n_repeats=3):
    """Importance par permutation, groupée par variable d'origine (toutes ses modalités permutées ensemble)."""
    key = (name, "perm")
    if key in sess.cache:
        return sess.cache[key]
    rs = np.random.RandomState(0)
    n = len(sess.M_test)
    cap = 500 if CATALOG[name]["kind"] == "knn" else 1500
    idx = rs.choice(n, min(n, cap), replace=False)
    y = sess.y_test.values[idx]
    if y.min() == y.max():
        idx = np.arange(n)
        y = sess.y_test.values
    M = sess.M_test[idx]
    base = roc_auc_score(y, proba(sess, name, sess.scaler.transform(M)))
    groups = {}
    for j, m in enumerate(sess.meta):
        groups.setdefault(m["orig"], []).append(j)
    items = []
    for orig, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rs.permutation(len(idx))
            M2 = M.copy()
            M2[:, cols] = M[perm][:, cols]
            drops.append(base - roc_auc_score(y, proba(sess, name, sess.scaler.transform(M2))))
        items.append({"feature": orig, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    items.sort(key=lambda d: -d["importance"])
    res = {"base_auc": float(base), "metric": "baisse d'AUC-ROC", "items": items}
    sess.cache[key] = res
    return res


# ---------------------------------------------------------------- explications locales
def _lime_explainer(sess):
    if "lime" in sess.cache:
        return sess.cache["lime"]
    n = len(sess.M_train)
    idx = np.random.RandomState(0).choice(n, min(n, 3000), replace=False)
    cat_idx = [j for j, m in enumerate(sess.meta) if m["kind"] == "cat"]
    ex = LimeTabularExplainer(
        sess.M_train[idx], feature_names=sess.names, categorical_features=cat_idx,
        categorical_names={j: ["0", "1"] for j in cat_idx},
        class_names=["classe 0", "classe 1"], discretize_continuous=True, random_state=0, mode="classification")
    sess.cache["lime"] = ex
    return ex


def _defaults(sess):
    X = sess.X_train_raw
    d = {}
    for c in sess.features:
        s = X[c]
        d[c] = s.median() if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) \
            else s.astype(str).mode().iloc[0]
    return d


def sentence(sess, p, thr, items, label_pos):
    """Phrase de synthèse à partir des contributions SHAP (items triés par |contribution|)."""
    pos = [i for i in items if i["shap"] > 0][:2]
    neg = [i for i in items if i["shap"] < 0][:2]
    if p >= thr:
        level = "élevé"
    elif p >= thr / 2:
        level = "modéré"
    else:
        level = "faible"
    head = f"Le risque « {label_pos} » est {level} (probabilité {p:.0%}, seuil {thr:.0%})"
    join = lambda xs: " et ".join(x["label"] for x in xs)
    if level == "faible":
        s = head + (f", principalement grâce à {join(neg)}" if neg else "")
        if pos:
            s += f", même si {join(pos[:1])} tire le risque vers le haut"
    else:
        s = head + (f", principalement à cause de {join(pos)}" if pos else ", sans facteur dominant")
        if neg:
            s += f" ; à l'inverse, {join(neg[:1])} le réduit"
    return s + "."


def local(sess, name, thr, test_index=None, record=None, lime_samples=1500, top=10):
    est = sess.models[name]["est"]
    true_label = None
    if record is not None:
        d = _defaults(sess)
        row = pd.DataFrame([{f: (record[f] if record.get(f) not in (None, "") else d[f]) for f in sess.features}])
        for f in sess.features:
            if pd.api.types.is_numeric_dtype(sess.X_train_raw[f]) and not pd.api.types.is_bool_dtype(sess.X_train_raw[f]):
                row[f] = pd.to_numeric(row[f], errors="coerce")
        M, Z = row_to_matrices(sess, row)
        raw = row.iloc[0].to_dict()
    else:
        i = int(test_index)
        if not 0 <= i < len(sess.M_test):
            raise ValueError("Indice hors du jeu de test.")
        M, Z = sess.M_test[[i]], sess.Z_test[[i]]
        raw = sess.X_test_raw.iloc[i].to_dict()
        true_label = int(sess.y_test.iloc[i])
    m_row = M[0]
    p = float(est.predict_proba(Z)[0, 1])

    # --- SHAP
    sv, base = shap_values(sess, name, Z)
    sv = sv[0]
    order = np.argsort(-np.abs(sv))
    items = [{"feature": sess.names[j], "label": col_label(sess, int(j), m_row), "shap": float(sv[j])}
             for j in order[:top]]
    others = float(sv[order[top:]].sum()) if len(order) > top else 0.0
    waterfall = {"base": base, "space": shap_space(name), "items": items, "others": others,
                 "n_others": int(max(0, len(order) - top)), "output": float(base + sv.sum())}

    # --- LIME
    lex = _lime_explainer(sess)
    exp = lex.explain_instance(m_row, lambda A: est.predict_proba(sess.scaler.transform(A)),
                               num_features=top, num_samples=lime_samples, labels=(1,))
    lmap = exp.as_map()[1]
    rules = [r for r, _ in exp.as_list(label=1)]
    lime_items = []
    for (j, w), rule in zip(lmap, rules):
        lbl = col_label(sess, int(j), m_row) if sess.meta[j]["kind"] == "cat" else rule
        lime_items.append({"feature": sess.names[j], "label": lbl, "weight": float(w), "idx": int(j)})

    # --- comparaison SHAP / LIME
    shap_top = {int(j): float(sv[j]) for j in order[:top]}
    lime_w = {it["idx"]: it["weight"] for it in lime_items}
    union = list(dict.fromkeys(list(shap_top) + list(lime_w)))
    comp = [{"feature": sess.names[j], "label": col_label(sess, j, m_row),
             "shap": float(sv[j]), "lime": lime_w.get(j),
             "same_sign": (None if j not in lime_w else bool(np.sign(sv[j]) == np.sign(lime_w[j])))}
            for j in union]
    both = [c for c in comp if c["lime"] is not None]
    sign_agree = float(np.mean([c["same_sign"] for c in both])) if both else None
    rho = None
    if len(both) >= 3:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = spearmanr([c["shap"] for c in both], [c["lime"] for c in both])[0]
        rho = None if r != r else float(r)
    top5s = set(list(shap_top)[:5])
    top5l = set([it["idx"] for it in lime_items][:5])
    agreement = {"top5_overlap": len(top5s & top5l), "sign_agreement": sign_agree, "spearman": rho,
                 "n_common": len(both)}

    return {
        "proba": p, "threshold": float(thr), "decision": bool(p >= thr), "true_label": true_label,
        "record": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in raw.items()},
        "waterfall": waterfall, "lime": {"items": lime_items, "intercept": float(exp.intercept[1]),
                                          "local_pred": float(exp.local_pred[0]) if hasattr(exp, "local_pred") else None},
        "comparison": comp, "agreement": agreement,
        "sentence": sentence(sess, p, thr, items, f"{sess.target} = {sess.positive}"),
    }


# ---------------------------------------------------------------- équité
_SENS = {"gender", "sex", "age", "senior", "citizen", "seniorcitizen", "race", "ethnic", "ethnicity",
         "nationality", "marital", "married", "partner", "dependents", "religion", "native", "origin",
         "country", "relationship", "marriage"}


def is_sensitive(col: str) -> bool:
    low = col.lower()
    if low in _SENS:
        return True
    toks = [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", col)]
    return any(t in _SENS for t in toks)


def sensitive_audit(sess, name, thr):
    p = proba(sess, name, sess.Z_test)
    y = sess.y_test.values
    shap_g = shap_global(sess, name)
    share = shap_g["share_by_orig"]
    out = []
    for c in sess.df.columns:
        if c == sess.target or not is_sensitive(c):
            continue
        s = sess.df.loc[sess.X_test_raw.index, c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 6:
            g = pd.qcut(s, 4, duplicates="drop").astype(str)
        else:
            g = s.astype(object).where(s.notna(), "(manquant)").astype(str)
        d = pd.DataFrame({"g": g.values, "p": p, "y": y})
        groups = []
        for k, sub in d.groupby("g"):
            pos = sub[sub.y == 1]
            groups.append({
                "group": k, "n": int(len(sub)), "actual_rate": float(sub.y.mean()),
                "mean_proba": float(sub.p.mean()), "flagged_rate": float((sub.p >= thr).mean()),
                "recall": float((pos.p >= thr).mean()) if len(pos) else None,
            })
        big = [g_["flagged_rate"] for g_ in groups if g_["n"] >= 20]
        out.append({
            "feature": c, "used_in_model": c in sess.features,
            "shap_share": float(share.get(c, 0.0)) if c in sess.features else 0.0,
            "parity_gap": float(max(big) - min(big)) if len(big) >= 2 else None,
            "groups": groups,
        })
    return {"threshold": float(thr), "features": out}
