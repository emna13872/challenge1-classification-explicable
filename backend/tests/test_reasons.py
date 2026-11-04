"""Motifs de refus : lisibles, honnêtes, jamais discriminatoires.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import models as MD
from app import prep as P
from app import reasons as R
from app.main import app
from app.state import Session


def _credit(n=2500, seed=0) -> pd.DataFrame:
    """L'âge pèse très fort sur le risque (biais volontaire) : il ne doit pourtant JAMAIS apparaître comme motif."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 70, n)
    sex = rng.choice(["F", "M"], n)
    debt = rng.gamma(2.0, 2000, n)
    paid = rng.gamma(2.0, 1500, n)
    job = rng.choice(["employe", "independant", "sans_emploi"], n)
    logit = -1.0 + 0.00035 * debt - 0.0005 * paid + 0.9 * (job == "sans_emploi") + 0.06 * (age - 45)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"age": age, "sex": sex, "debt": debt, "paid": paid, "job": job, "y": y.astype(str)})


def _trained(model="xgb"):
    s = Session()
    s.df, s.target, s.positive, s.name = _credit(), "y", "1", "synthetic"
    P.build(s, {"features": ["age", "sex", "debt", "paid", "job"], "test_size": 0.25, "random_state": 1,
                "num_impute": "median", "cat_impute": "mode", "scaling": "standard", "stratify": True})
    est, info = MD.train_one(model, {}, False, False, s.Z_train, s.y_train.values)
    s.models[model] = {"est": est, "info": info, "p_test": est.predict_proba(s.Z_test)[:, 1], "label": model}
    return s


def _riskiest(s, model, k=5):
    return [int(i) for i in np.argsort(-s.models[model]["p_test"])[:k]]


@pytest.mark.parametrize("model", ["xgb", "logreg"])
def test_contributions_add_up_to_the_prediction(model):
    s = _trained(model)
    for i in _riskiest(s, model) + [0, 1, 2]:
        r = R.decision_reasons(s, model, 0.3, test_index=i)
        assert r["additivity_gap"] < 1e-3, (model, i, r["additivity_gap"])


def test_sensitive_variables_are_never_a_reason_even_when_they_dominate():
    s = _trained("xgb")
    seen_excluded = False
    for i in _riskiest(s, "xgb", 25):
        r = R.decision_reasons(s, "xgb", 0.3, test_index=i, top=10)
        named = {x["feature"] for x in r["reasons_against"] + r["reasons_for"]}
        assert not named & {"age", "sex"}
        seen_excluded |= {"age", "sex"} <= set(r["excluded_sensitive"])
    assert seen_excluded                                                  # elles sont bien dans le modèle, mais cachées


def test_reasons_are_sorted_grouped_and_limited():
    s = _trained("xgb")
    r = R.decision_reasons(s, "xgb", 0.3, test_index=_riskiest(s, "xgb")[0], top=3)
    contribs = [x["contribution"] for x in r["reasons_against"]]
    assert len(contribs) <= 3 and contribs == sorted(contribs, reverse=True) and all(c > 0 for c in contribs)
    assert all(x["contribution"] < 0 for x in r["reasons_for"])
    feats = [x["feature"] for x in r["reasons_against"] + r["reasons_for"]]
    assert len(feats) == len(set(feats)) and "job" not in {f for f in feats if "_" in f}   # « job » = un motif, pas 3 colonnes


def test_decision_follows_the_threshold():
    s = _trained("xgb")
    pt = s.models["xgb"]["p_test"]
    i = int(np.argmin(np.abs(pt - 0.5)))                                 # un client au milieu : ni quasi sûr, ni quasi perdu
    p = R.decision_reasons(s, "xgb", 0.5, test_index=i)["probability"]
    assert 0.1 < p < 0.9
    assert R.decision_reasons(s, "xgb", p + 0.01, test_index=i)["decision"] == "accord"
    assert R.decision_reasons(s, "xgb", p - 0.01, test_index=i)["decision"] == "refus"


def test_texts_contain_the_real_numbers_and_are_deterministic():
    s = _trained("xgb")
    i = _riskiest(s, "xgb")[0]
    a = R.decision_reasons(s, "xgb", 0.3, test_index=i)
    b = R.decision_reasons(s, "xgb", 0.3, test_index=i)
    assert a == b
    top = a["reasons_against"][0]
    if top["kind"] == "num":
        assert f"{top['value']:.0f}" in top["text"] or f"{top['value']:.1f}" in top["text"] or "." in top["text"]
        assert "médiane" in top["text"]
    assert top["feature"] in ("debt", "paid", "job")


def test_works_for_a_made_up_client_and_rejects_bad_input():
    s = _trained("xgb")
    r = R.decision_reasons(s, "xgb", 0.3, record={"debt": 20000, "paid": 100, "job": "sans_emploi", "age": 30, "sex": "F"})
    assert r["decision"] == "refus" and r["reasons_against"]
    with pytest.raises(ValueError):
        R.decision_reasons(s, "xgb", 0.3, test_index=10**6)
    with pytest.raises(ValueError):
        R.decision_reasons(s, "xgb", 1.5, test_index=0)
    with pytest.raises(ValueError):
        R.decision_reasons(s, "rf", 0.3, test_index=0)                    # modèle non entraîné


def test_reasons_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-reasons"}
    buf = io.StringIO()
    _credit().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("credit.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["age", "sex", "debt", "paid", "job"]}, headers=H).status_code == 200
    assert c.post("/api/models/train", json={"models": [{"name": "xgb"}]}, headers=H).status_code == 200
    r = c.post("/api/decision/reasons", json={"model": "xgb", "threshold": 0.3, "test_index": 0}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["decision"] in ("refus", "accord") and r.json()["additivity_gap"] < 1e-3
    assert c.post("/api/decision/reasons", json={"model": "xgb"}, headers=H).status_code == 400
    assert c.post("/api/decision/reasons", json={"model": "rf", "test_index": 0}, headers=H).status_code == 404
    assert c.post("/api/decision/reasons", json={"model": "xgb", "test_index": 10**6}, headers=H).status_code == 400


def test_taiwan_repayment_codes_are_readable():
    s = Session()
    s.name = "taiwan_credit"
    assert R._fmt_value(s, "PAY_1", 2) == "retard de 2 mois"
    assert R._fmt_value(s, "PAY_3", -1) == "payé à temps"
    assert "non documenté" in R._fmt_value(s, "PAY_2", 0)
    assert R._fmt_value(s, "PAY_AMT1", 291) == "291"                    # un montant n'est pas un code
    s.name = "autre"
    assert R._fmt_value(s, "PAY_1", 2) == "2"                           # aucun code supposé sur un autre jeu
