"""Fraude (phase C) : décider quand les cas positifs sont très rares et que chaque erreur a un prix DIFFÉRENT.

Trois idées.
1. L'exactitude trompe : « tout est légitime » a 99,8 % d'exactitude et n'attrape aucune fraude. On mesure la
   PR-AUC (aire sous la courbe précision-rappel) et ce qu'on attrape pour un budget d'alertes donné.
2. On valide DANS LE TEMPS : apprendre sur le passé, noter le futur (validation glissante, comme en production).
   Le lot de test est le dernier quart des transactions (découpage temporel du module 2) et s'ouvre une fois.
3. Une fraude manquée coûte SON montant ; une alerte coûte une vérification (hypothèse : `review_cost`).
   On alerte si p × montant > coût de vérification : une fraude probable de 2 € ne vaut pas une vérification de 5 €,
   une fraude possible de 500 € oui. Cette règle n'est juste que si p est une vraie probabilité (calibration).

Coût total d'une stratégie = montants des fraudes laissées passer + coût de vérification × nombre d'alertes.
Ce module ne lit le lot de test que dans `fraud_final`, une fois, et chaque ouverture est comptée.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from . import models as MD
from .prep import ScaleFirst
from .state import Session

DEFAULT_VARIANTS = [("logreg", False), ("logreg", True), ("xgb", False), ("xgb", True)]
BUDGETS = (0.5, 1.0, 2.0, 5.0)          # alertes pour 1 000 transactions
STRATEGY_LABELS = {
    "accept_all": "Ne rien vérifier",
    "half": "Seuil fixe de 50 %",
    "best_threshold": "Meilleur seuil unique (choisi sur l'apprentissage)",
    "amount_rule": "Règle du montant : alerter si p × montant > coût de vérification",
}


def _key(name: str, balance: bool) -> str:
    return f"{name}_balanced" if balance else name


def _label(name: str, balance: bool) -> str:
    return MD.CATALOG[name]["label"] + (" (classes rééquilibrées)" if balance else "")


def _amounts(sess: Session, index, amount_column: str) -> np.ndarray:
    if amount_column not in sess.df.columns:
        raise ValueError(f"Colonne de montant « {amount_column} » introuvable : la décision par le coût en a besoin.")
    return sess.df.loc[index, amount_column].to_numpy(float)


def forward_oof(sess: Session, name: str, balance: bool, n_splits: int = 4):
    """Probabilités « hors-pli » sur l'apprentissage, sans jamais utiliser le futur pour noter le passé.

    Découpage temporel : les transactions d'apprentissage, dans l'ordre du temps, sont coupées en n_splits + 1 blocs ;
    le bloc k est noté par un modèle appris sur les blocs 1 à k-1 (le premier bloc n'est jamais noté).
    Sans colonne de temps : validation croisée stratifiée ordinaire (signalé dans le résultat).
    Retourne (positions notées dans le lot d'apprentissage, y, p, temporel ?).
    """
    X, y = sess.X_train_raw, np.asarray(sess.y_train)
    tcol = sess.prep_cfg.get("time_column")
    if tcol:
        order = np.argsort(sess.df.loc[X.index, tcol].to_numpy(float), kind="mergesort")
        splits = [(order[tr], order[va]) for tr, va in TimeSeriesSplit(n_splits=n_splits).split(order)]
    else:
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y))
    pos, ps = [], []
    for tr, va in splits:
        imp = clone(sess.imp)                                         # préparation réapprise sur le passé seul
        Mtr = imp.fit_transform(X.iloc[tr]).astype(float)
        Mva = imp.transform(X.iloc[va]).astype(float)
        sc = ScaleFirst(sess.prep_cfg.get("scaling", "standard"), sess.n_num).fit(Mtr)
        ytr = y[tr]
        if ytr.sum() == 0:
            raise ValueError("Un bloc d'apprentissage ne contient aucune fraude : réduisez le nombre de blocs.")
        est = MD.make(name, None, balance, float((ytr == 0).sum() / ytr.sum()))
        est.fit(sc.transform(Mtr), ytr)
        pos.append(va)
        ps.append(est.predict_proba(sc.transform(Mva))[:, 1])
    pos = np.concatenate(pos)
    return pos, y[pos], np.concatenate(ps), bool(tcol)


def _outcome(y, A, flag, c) -> dict:
    """Bilan d'une stratégie : ce qu'on attrape, ce qu'on laisse passer, ce que coûtent les vérifications."""
    y, A, flag = np.asarray(y), np.asarray(A, float), np.asarray(flag, bool)
    n, n_fraud = len(y), int(y.sum())
    total = float(A[y == 1].sum())
    caught = int((flag & (y == 1)).sum())
    alerts = int(flag.sum())
    amount_caught = float(A[flag & (y == 1)].sum())
    cost = (total - amount_caught) + c * alerts
    return {"alerts": alerts, "alerts_per_1000": 1000 * alerts / n, "frauds_caught": caught, "frauds_total": n_fraud,
            "recall": caught / n_fraud if n_fraud else None, "precision": caught / alerts if alerts else None,
            "amount_caught": amount_caught, "amount_missed": total - amount_caught, "review_cost": c * alerts,
            "total_cost": cost, "cost_per_1000": 1000 * cost / n,
            "saved": total - cost, "saved_share": (total - cost) / total if total else None}


def best_threshold(y, p, A, c):
    """Le seuil unique qui minimise le coût : on classe par p décroissant et on cherche le meilleur nombre d'alertes."""
    order = np.argsort(-np.asarray(p), kind="mergesort")
    gain = np.cumsum((np.asarray(A, float) * np.asarray(y))[order])               # montants attrapés
    k = np.arange(1, len(order) + 1)
    cost = (gain[-1] - gain) + c * k
    i = int(np.argmin(cost))
    if gain[-1] <= cost[i]:                                                         # mieux vaut ne rien vérifier
        return 1.0 + 1e-9
    return float(np.asarray(p)[order][i])


def strategies(y, p, A, c, thr: float | None = None, budgets=BUDGETS) -> dict:
    """Toutes les stratégies sur les mêmes transactions ; `thr` = seuil choisi ailleurs (sinon : optimisé ici)."""
    p, A = np.asarray(p), np.asarray(A, float)
    thr = best_threshold(y, p, A, c) if thr is None else thr
    out = {
        "accept_all": _outcome(y, A, np.zeros(len(p), bool), c),
        "half": _outcome(y, A, p >= 0.5, c),
        "best_threshold": {**_outcome(y, A, p >= thr, c), "threshold": thr},
        "amount_rule": _outcome(y, A, p * A > c, c),
    }
    order = np.argsort(-p, kind="mergesort")
    out["budgets"] = []
    for b in budgets:
        k = int(round(len(p) * b / 1000))
        flag = np.zeros(len(p), bool)
        flag[order[:k]] = True
        out["budgets"].append({"per_1000": b, **_outcome(y, A, flag, c)})
    return out


def calibration_summary(y, p) -> dict:
    """La probabilité moyenne annoncée tient-elle sa promesse ? (global, et pour le 1 % le plus suspect)."""
    y, p = np.asarray(y), np.asarray(p)
    top = np.argsort(-p)[: max(1, len(p) // 100)]
    return {"mean_predicted": float(p.mean()), "base_rate": float(y.mean()),
            "ratio": float(p.mean() / y.mean()) if y.mean() else None,
            "top1pct_mean_predicted": float(p[top].mean()), "top1pct_rate": float(y[top].mean())}


def fraud_report(sess: Session, variants=None, review_cost: float = 5.0, n_splits: int = 4,
                 amount_column: str = "Amount") -> dict:
    """Compare les variantes de modèles et les stratégies, sur l'apprentissage seul (validation glissante)."""
    if sess.X_train_raw is None:
        raise ValueError("Préparez d'abord les données (module 2).")
    if review_cost <= 0:
        raise ValueError("Le coût d'une vérification doit être positif.")
    variants = [tuple(v) for v in (variants or DEFAULT_VARIANTS)]
    rows, temporal, n_scored, n_fraud_scored = [], False, 0, 0
    for name, bal in variants:
        if name not in MD.CATALOG:
            raise ValueError(f"Modèle inconnu : {name}")
        pos, y, p, temporal = forward_oof(sess, name, bal, n_splits)
        A = _amounts(sess, sess.X_train_raw.index[pos], amount_column)
        st = strategies(y, p, A, review_cost)
        n_scored, n_fraud_scored = len(y), int(y.sum())
        rows.append({"key": _key(name, bal), "model": name, "balance": bal, "label": _label(name, bal),
                     "pr_auc": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p)),
                     "accuracy_accept_all": float(1 - y.mean()),
                     "calibration": calibration_summary(y, p), "strategies": st})
    cands = [(r["strategies"][s]["total_cost"], r["key"], s) for r in rows for s in ("best_threshold", "amount_rule")]
    cost, key, strat = min(cands)
    chosen = next(r for r in rows if r["key"] == key)
    choice = {"key": key, "model": chosen["model"], "balance": chosen["balance"], "label": chosen["label"],
              "strategy": strat, "strategy_label": STRATEGY_LABELS[strat],
              "threshold": chosen["strategies"]["best_threshold"]["threshold"] if strat == "best_threshold" else None,
              "expected": chosen["strategies"][strat]}
    sess.fraud_choice = {**choice, "review_cost": review_cost, "amount_column": amount_column}
    return {"variants": rows, "choice": choice, "review_cost": review_cost,
            "strategy_labels": STRATEGY_LABELS,
            "settings": {"temporal": temporal, "n_splits": n_splits, "n_scored": n_scored,
                         "n_frauds_scored": n_fraud_scored, "amount_column": amount_column},
            "test_set_used": False}


def _bootstrap_cost_ci(y, A, flag, c, reps: int = 400, seed: int = 0) -> list[float]:
    """Intervalle à 95 % du coût pour 1 000 transactions (rééchantillonnage des transactions du test)."""
    y, A, flag = np.asarray(y), np.asarray(A, float), np.asarray(flag, bool)
    per = c * flag + A * y * (~flag)                                                # coût de chaque transaction
    rng = np.random.default_rng(seed)
    n = len(per)
    vals = [1000 * per[rng.integers(0, n, n)].mean() for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fraud_final(sess: Session, review_cost: float = 5.0, variants=None, n_splits: int = 4,
                amount_column: str = "Amount", force: bool = False) -> dict:
    """Décide tout sur l'apprentissage (fraud_report), puis ouvre le lot de test UNE fois pour vérifier la promesse."""
    if sess.Z_test is None:
        raise ValueError("Préparez d'abord les données (module 2).")
    if sess.fraud_final is not None and not force:
        return {**sess.fraud_final, "already_opened": True}
    rep = fraud_report(sess, variants, review_cost, n_splits, amount_column)
    ch = rep["choice"]
    ytr = np.asarray(sess.y_train)
    est = MD.make(ch["model"], None, ch["balance"], float((ytr == 0).sum() / max(1, ytr.sum())))
    est.fit(sess.Z_train, ytr)

    sess.test_openings += 1                                                         # ouverture comptée
    yt = np.asarray(sess.y_test)
    p = est.predict_proba(sess.Z_test)[:, 1]
    A = _amounts(sess, sess.X_test_raw.index, amount_column)
    row = next(r for r in rep["variants"] if r["key"] == ch["key"])
    thr = row["strategies"]["best_threshold"]["threshold"]                         # seuil choisi sur l'apprentissage
    st = strategies(yt, p, A, review_cost, thr=thr)                                 # rien n'est ré-optimisé sur le test
    flag = (p >= thr) if ch["strategy"] == "best_threshold" else (p * A > review_cost)
    got = _outcome(yt, A, flag, review_cost)
    ci = _bootstrap_cost_ci(yt, A, flag, review_cost)
    res = {
        "choice": ch, "review_cost": review_cost,
        "expected": ch["expected"],                                                 # la promesse de la validation
        "test": {"n": int(len(yt)), "base_rate": float(yt.mean()), "pr_auc": float(average_precision_score(yt, p)),
                 "auc": float(roc_auc_score(yt, p)), "accuracy_accept_all": float(1 - yt.mean()),
                 "chosen": got, "cost_per_1000_ci95": ci,
                 "expected_inside_ci": bool(ci[0] <= ch["expected"]["cost_per_1000"] <= ci[1]),
                 "accept_all": st["accept_all"], "half": st["half"], "amount_rule": st["amount_rule"],
                 "train_threshold": st["best_threshold"],
                 "budgets": st["budgets"], "calibration": calibration_summary(yt, p)},
        "train_base_rate": float(ytr.mean()),
        "settings": rep["settings"], "test_set_used": True, "openings": sess.test_openings, "already_opened": False,
        "warning": None if sess.test_openings == 1 else
                   f"Le lot de test a été ouvert {sess.test_openings} fois : cette note n'est plus tout à fait honnête.",
    }
    sess.fraud_final = res
    return res
