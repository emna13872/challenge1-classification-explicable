"""Fraude (phase C) : découpage temporel, bilans exacts, règle du montant, test ouvert une fois, pas de conseil.

Données synthétiques fabriquées ici (même forme que le jeu ULB : Time, V1…, Amount, Class) : tournent partout.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import fraud as F
from app import models as MD
from app import prep as P
from app import reasons as R
from app import recourse as RC
from app import recourse_audit as RA
from app.main import app
from app.state import Session

FEATS = ["V1", "V2", "V3", "V4", "Amount"]


def _fraud(n=8000, seed=0) -> pd.DataFrame:
    """Environ 1,5 % de fraudes ; elles se repèrent par V1 et V2 ; leurs montants sont plus étalés."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 172800, n))                         # 48 heures, en secondes
    y = (rng.random(n) < 0.015).astype(int)
    V = rng.normal(size=(n, 4))
    V[:, 0] += 2.2 * y
    V[:, 1] -= 1.8 * y
    amount = np.where(y == 1, rng.choice([1.0, 9.0, 150.0, 600.0], n), rng.gamma(1.2, 60, n)).round(2)
    return pd.DataFrame({"Time": t, "V1": V[:, 0], "V2": V[:, 1], "V3": V[:, 2], "V4": V[:, 3],
                         "Amount": amount, "Class": y.astype(str)})


def _session(name="synthetic_fraud", temporal=True):
    s = Session()
    s.df, s.target, s.positive, s.name = _fraud(), "Class", "1", name
    P.build(s, {"features": FEATS, "test_size": 0.25, "num_impute": "median", "cat_impute": "mode",
                "scaling": "standard", **({"time_column": "Time"} if temporal else {})})
    return s


@pytest.fixture(scope="module")
def s():
    return _session()


VARIANTS = [("logreg", False), ("logreg", True), ("xgb", False)]


def test_temporal_split_tests_on_the_future(s):
    t_tr = s.df.loc[s.X_train_raw.index, "Time"]
    t_te = s.df.loc[s.X_test_raw.index, "Time"]
    assert t_tr.max() < t_te.min()                                 # tout le test est après tout l'apprentissage
    assert len(t_te) == round(0.25 * len(s.df))
    r = _session(temporal=False)                                   # par défaut : découpage au hasard, inchangé
    assert r.df.loc[r.X_train_raw.index, "Time"].max() > r.df.loc[r.X_test_raw.index, "Time"].min()


def test_temporal_split_rejects_a_bad_time_column():
    s = Session()
    s.df, s.target, s.positive, s.name = _fraud(), "Class", "1", "x"
    with pytest.raises(ValueError, match="temps"):
        P.build(s, {"features": FEATS, "time_column": "absente"})


def test_forward_validation_never_scores_the_oldest_block(s):
    pos, y, p, temporal = F.forward_oof(s, "logreg", False, n_splits=4)
    assert temporal and len(np.unique(pos)) == len(pos)            # chaque transaction notée une fois au plus
    t = s.df.loc[s.X_train_raw.index, "Time"].to_numpy()
    scored = np.zeros(len(t), bool)
    scored[pos] = True
    assert t[~scored].max() < t[scored].min()                      # le plus ancien bloc sert seulement à apprendre
    assert scored.mean() == pytest.approx(0.8, abs=0.01)           # 4 blocs notés sur 5


def test_outcome_on_a_hand_example():
    o = F._outcome(y=[1, 0, 1, 0], A=[100, 20, 3, 50], flag=[1, 1, 0, 0], c=5)
    assert (o["frauds_caught"], o["alerts"], o["amount_caught"], o["amount_missed"]) == (1, 2, 100, 3)
    assert o["review_cost"] == 10 and o["total_cost"] == 13 and o["saved"] == 90   # 103 de fraude, il en coûte 13
    assert o["recall"] == 0.5 and o["precision"] == 0.5


def test_best_threshold_and_amount_rule_on_a_hand_example():
    y, A = np.array([1, 0, 1, 0, 1]), np.array([200.0, 40.0, 2.0, 30.0, 80.0])
    p = np.array([0.9, 0.6, 0.5, 0.2, 0.1])
    # 1 alerte : coût 82+5 ; 2 : 82+10 ; 3 : 80+15 ; 4 : 80+20 ; 5 : 0+25 → le mieux est de tout vérifier (seuil 0,1)
    assert F.best_threshold(y, p, A, 5) == pytest.approx(0.1)
    assert F.best_threshold(y, p, A, 10 ** 6) > 1                  # vérifier coûte plus que toutes les fraudes
    st = F.strategies(y, p, A, 5)
    flags = p * A > 5                                              # 180, 24, 1, 6, 8 → la fraude de 2 n'est jamais vérifiée
    assert st["amount_rule"]["alerts"] == int(flags.sum()) == 4
    assert st["amount_rule"]["amount_missed"] == 2


def test_rebalancing_inflates_probabilities_and_breaks_the_amount_rule(s):
    rep = F.fraud_report(s, VARIANTS, review_cost=5)
    by = {v["key"]: v for v in rep["variants"]}
    assert by["logreg_balanced"]["calibration"]["ratio"] > 5       # probabilités gonflées
    assert 0.5 < by["logreg"]["calibration"]["ratio"] < 2          # à peu près justes
    assert (by["logreg_balanced"]["strategies"]["amount_rule"]["alerts"]
            > 3 * by["logreg"]["strategies"]["amount_rule"]["alerts"])
    assert rep["choice"]["strategy"] in ("best_threshold", "amount_rule") and not rep["test_set_used"]


def test_the_choice_never_reads_the_test_set(s):
    a = F.fraud_report(s, VARIANTS, review_cost=5)
    saved = s.y_test.copy()
    s.y_test = 1 - s.y_test                                        # sabotage : on inverse les étiquettes du test
    try:
        b = F.fraud_report(s, VARIANTS, review_cost=5)
    finally:
        s.y_test = saved
    assert a["choice"] == b["choice"] and a["variants"] == b["variants"]


def test_test_set_is_opened_once_and_counted():
    s = _session()
    f = F.fraud_final(s, review_cost=5, variants=VARIANTS)
    assert f["openings"] == 1 and not f["already_opened"] and f["test_set_used"]
    assert f["test"]["chosen"]["frauds_total"] == int(s.y_test.sum())
    lo, hi = f["test"]["cost_per_1000_ci95"]
    assert lo <= f["test"]["chosen"]["cost_per_1000"] <= hi
    again = F.fraud_final(s, review_cost=5, variants=VARIANTS)
    assert again["already_opened"] and s.test_openings == 1        # pas de seconde lecture
    forced = F.fraud_final(s, review_cost=5, variants=VARIANTS, force=True)
    assert forced["openings"] == 2 and forced["warning"]
    P.build(s, s.prep_cfg)                                         # nouveau découpage : compteur remis à zéro
    assert s.test_openings == 0 and s.fraud_final is None


def test_invalid_inputs(s):
    with pytest.raises(ValueError):
        F.fraud_report(s, VARIANTS, review_cost=0)
    with pytest.raises(ValueError, match="montant"):
        F.fraud_report(s, VARIANTS, amount_column="Montant")
    with pytest.raises(ValueError):
        F.fraud_report(s, [("inconnu", False)])


def test_no_advice_and_honest_reasons_for_fraud():
    s = _session(name="ulb_fraud")
    est, info = MD.train_one("xgb", {}, False, False, s.Z_train, s.y_train.values)
    s.models["xgb"] = {"est": est, "info": info, "label": "xgb"}
    r = RC.recourse(s, "xgb", 0.1, test_index=0)
    assert r["status"] == "not_applicable" and "contourner" in r["reason"] and r["options"] == []
    assert RA.audit(s, "xgb", 0.1)["status"] == "not_applicable"
    assert "anonymisées" in R.decision_reasons(s, "xgb", 0.1, test_index=0)["note"]
    s.name = "autre"
    assert R.decision_reasons(s, "xgb", 0.1, test_index=0)["note"] is None


def test_fraud_endpoints():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-fraud"}
    buf = io.StringIO()
    _fraud().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("f.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "Class"}, headers=H).status_code == 200
    assert c.post("/api/fraud/report", json={}, headers=H).status_code == 400          # pas encore préparé
    p = c.post("/api/prep", json={"features": FEATS, "time_column": "Time"}, headers=H)
    assert p.status_code == 200 and p.json()["split"] == "temporal"
    assert c.post("/api/prep", json={"features": FEATS, "time_column": "V9"}, headers=H).status_code == 400
    assert c.post("/api/prep", json={"features": FEATS, "time_column": "Time"}, headers=H).status_code == 200
    body = {"review_cost": 5, "variants": [["logreg", False], ["xgb", False]]}
    r = c.post("/api/fraud/report", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["temporal"]
    assert c.post("/api/fraud/report", json={**body, "review_cost": 0}, headers=H).status_code == 422
    f = c.post("/api/fraud/final", json=body, headers=H)
    assert f.status_code == 200 and f.json()["openings"] == 1
