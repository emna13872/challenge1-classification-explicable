"""Audit du recours (B4) : les comptes sont justes, l'audit dit la même chose que le conseil client par client,
et une inégalité d'accès au conseil entre groupes est bien détectée.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import models as MD
from app import prep as P
from app import recourse as RC
from app import recourse_audit as RA
from app.main import app
from app.state import Session
from test_reasons import _credit, _trained

RULES = {"paid": "up", "debt": "down"}
THR = 0.25


@pytest.fixture(scope="module")
def s():
    """Un seul entraînement pour tout le fichier (les tests ne modifient pas la session, sauf copie explicite)."""
    return _trained("xgb")


@pytest.fixture(scope="module")
def full(s):
    """L'audit complet (tous les clients refusés), calculé une fois et relu par plusieurs tests."""
    return RA.audit(s, "xgb", THR, max_clients=None, actionable=RULES)


def _refused(s, thr=THR):
    return np.where(s.models["xgb"]["est"].predict_proba(s.Z_test)[:, 1] >= thr)[0]


def test_counts_add_up_and_match_the_number_of_refused_clients(s, full):
    a = full
    assert a["status"] == "ok" and not a["sampled"]
    assert a["n_refused"] == len(_refused(s)) == a["n_audited"]
    assert a["n_found"] + a["n_none"] == a["n_audited"]
    assert abs(a["share_found"] + a["share_none"] - 1) < 1e-9
    assert abs(a["share_refused"] - a["n_refused"] / a["n_test"]) < 1e-9
    assert a["n_unverified"] == 0                                    # tout conseil compté a été re-vérifié


def test_audit_agrees_with_the_client_by_client_recourse(s):
    """L'audit ne refait pas un calcul « à côté » : il compte exactement ce que verrait chaque client."""
    a = RA.audit(s, "xgb", THR, max_clients=25, actionable=RULES)
    idx = np.sort(np.random.default_rng(0).choice(_refused(s), 25, replace=False))   # même tirage que l'audit
    found = sum(RC.recourse(s, "xgb", THR, test_index=int(i), actionable=RULES)["status"] == "found" for i in idx)
    assert found == a["n_found"]


def test_sampling_is_reproducible_and_says_so(s):
    a = RA.audit(s, "xgb", THR, max_clients=20, actionable=RULES)
    b = RA.audit(s, "xgb", THR, max_clients=20, actionable=RULES)
    assert a["sampled"] and a["n_audited"] == 20 and a["n_refused"] > 20
    a.pop("seconds"), b.pop("seconds")
    assert a == b                                                     # même graine : même résultat
    lo, hi = a["share_found_ci95"]
    assert lo <= a["share_found"] <= hi and hi - lo > 0.1             # 20 clients : l'incertitude est grande, et on la dit


def test_effort_numbers_are_coherent(full):
    e = full["effort"]
    assert 0 <= e["share_one_change"] <= 1 and 0 <= e["share_from_zero"] <= 1
    assert 0 < e["median_percentile_jump"] <= e["p90_percentile_jump"] <= 0.95
    assert set(e["variables_used"]) <= set(RULES)                    # jamais une variable non déclarée


def test_clients_without_advice_are_riskier_than_those_with_advice(full):
    """Ce qui distingue « aucun conseil » : un risque de départ plus haut, que les petits changements ne compensent pas."""
    a = full
    nf = a["none_found"]
    if a["n_none"] and a["n_found"]:
        assert nf["median_probability"] > nf["median_probability_when_found"]
        assert nf["median_lowest_probability"] >= THR                # sinon un conseil aurait été trouvé


def test_honest_statuses(s):
    assert RA.audit(s, "xgb", 0.9999, actionable=RULES)["status"] == "no_refused"
    old, s.name = s.name, "autre"                                    # rien n'est déclaré modifiable pour ce jeu
    try:
        assert RA.audit(s, "xgb", THR, max_clients=20)["status"] == "no_actionable"
    finally:
        s.name = old
    with pytest.raises(ValueError):
        RA.audit(s, "rf", THR)
    with pytest.raises(ValueError):
        RA.audit(s, "xgb", 1.5)


def test_wilson_interval():
    assert RA.wilson(0, 0) is None
    lo, hi = RA.wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35                              # 0 sur 10 ne prouve pas « jamais »
    assert RA.wilson(20, 20) == [pytest.approx(0.839, abs=1e-3), 1.0]   # 20 sur 20 ne prouve pas « toujours »
    lo, hi = RA.wilson(500, 1000)
    assert abs(lo - 0.469) < 0.002 and abs(hi - 0.531) < 0.002      # ± 3 points pour 1 000 clients


def _proxy_credit(n=3000, seed=3) -> pd.DataFrame:
    """Les femmes de ce jeu fictif sont plus souvent « sans_emploi » (variable que le client ne peut pas changer) :
    même sans que le sexe pèse directement, elles devraient recevoir MOINS de conseils. L'audit doit le voir."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    job = np.where(rng.random(n) < np.where(sex == "F", 0.6, 0.1), "sans_emploi", "employe")
    debt = rng.gamma(2.0, 2000, n)
    paid = rng.gamma(2.0, 1500, n)
    logit = -1.5 + 0.0003 * debt - 0.0004 * paid + 3.0 * (job == "sans_emploi")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"sex": sex, "debt": debt, "paid": paid, "job": job, "y": y.astype(str)})


def test_unequal_access_to_advice_is_detected_even_without_using_the_sensitive_variable():
    s = Session()
    s.df, s.target, s.positive, s.name = _proxy_credit(), "y", "1", "proxy"
    P.build(s, {"features": ["sex", "debt", "paid", "job"], "test_size": 0.3, "random_state": 0,
                "num_impute": "median", "cat_impute": "mode", "scaling": "standard", "stratify": True})
    est, info = MD.train_one("xgb", {}, False, False, s.Z_train, s.y_train.values)
    s.models["xgb"] = {"est": est, "info": info, "label": "xgb"}
    a = RA.audit(s, "xgb", 0.3, max_clients=None, actionable={"paid": "up"})
    sex = next(f for f in a["fairness"] if f["variable"] == "sex")
    share = {g["group"]: g["share_found"] for g in sex["groups"]}
    assert share["F"] < share["M"] and sex["alert"]
    assert all(g["n"] >= RA.MIN_GROUP for g in sex["groups"])


def test_small_groups_are_not_reported(full):
    a = full
    for f in a["fairness"]:
        assert all(g["n"] >= RA.MIN_GROUP for g in f["groups"]) and len(f["groups"]) >= 2
        assert all(g["ci95"][0] <= g["share_found"] <= g["ci95"][1] for g in f["groups"])
    assert {f["variable"] for f in a["fairness"]} <= {"age", "sex"}  # seules les variables sensibles sont auditées


def test_audit_endpoint():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-audit"}
    buf = io.StringIO()
    _credit().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("credit.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["age", "sex", "debt", "paid", "job"]}, headers=H).status_code == 200
    assert c.post("/api/models/train", json={"models": [{"name": "xgb"}]}, headers=H).status_code == 200
    body = {"model": "xgb", "threshold": 0.25, "max_clients": 30, "actionable": {"paid": "up"}}
    r = c.post("/api/decision/audit", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["n_audited"] == 30
    assert c.post("/api/decision/audit", json={**body, "actionable": {"age": "up"}}, headers=H).status_code == 400
    assert c.post("/api/decision/audit", json={**body, "model": "rf"}, headers=H).status_code == 404
    assert c.post("/api/decision/audit", json={**body, "threshold": 1.2}, headers=H).status_code == 422
