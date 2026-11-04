"""Recours contrefactuel : réaliste, vérifié, honnête quand il n'y a rien à proposer.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import explain as X
from app import recourse as RC
from app.main import app
from test_reasons import _credit, _trained

RULES = {"paid": "up", "debt": "down"}
THR = 0.25


def _refused_client(s, lo=0.3, hi=0.7):
    p = s.models["xgb"]["p_test"]
    ok = [i for i in range(len(p)) if lo < p[i] < hi]
    assert ok, "le jeu synthétique doit contenir un client refusé de risque moyen"
    return ok[0]


def test_every_option_really_flips_the_decision_through_the_full_pipeline():
    s = _trained("xgb")
    i = _refused_client(s)
    r = RC.recourse(s, "xgb", THR, test_index=i, actionable=RULES)
    assert r["status"] == "found" and r["probability"] >= THR
    raw = X.client_matrices(s, test_index=i)[2]
    for o in r["options"]:
        assert o["verified"] and o["new_probability"] < THR
        raw2 = dict(raw)
        for c in o["changes"]:
            raw2[c["feature"]] = c["to"]
        _, Z2, _, _ = X.client_matrices(s, record=raw2)
        assert float(s.models["xgb"]["est"].predict_proba(Z2)[0, 1]) < THR     # contrôle indépendant


def test_only_declared_variables_are_touched_and_values_stay_realistic():
    s = _trained("xgb")
    r = RC.recourse(s, "xgb", THR, test_index=_refused_client(s), actionable=RULES)
    X_tr = s.X_train_raw
    for o in r["options"]:
        assert 1 <= len(o["changes"]) <= 2
        for c in o["changes"]:
            assert c["feature"] in RULES
            lo, hi = np.percentile(X_tr[c["feature"]], [1, 99])
            assert lo - 1e-9 <= c["to"] <= hi + 1e-9                          # valeurs de vrais clients
            assert c["percentile_to"] <= 0.97 if RULES[c["feature"]] == "up" else c["percentile_to"] >= 0.03
            assert (c["to"] > c["from"]) if RULES[c["feature"]] == "up" else (c["to"] < c["from"])


def test_sensitive_or_unknown_variables_are_refused():
    s = _trained("xgb")
    with pytest.raises(ValueError, match="sensible"):
        RC.recourse(s, "xgb", THR, test_index=0, actionable={"age": "both"})
    with pytest.raises(ValueError, match="sensible"):
        RC.recourse(s, "xgb", THR, test_index=0, actionable={"sex": "both"})
    with pytest.raises(ValueError):
        RC.recourse(s, "xgb", THR, test_index=0, actionable={"inconnue": "up"})
    with pytest.raises(ValueError):
        RC.recourse(s, "xgb", THR, test_index=0, actionable={"paid": "sideways"})


def test_honest_answers_when_nothing_can_be_proposed():
    s = _trained("xgb")
    p = s.models["xgb"]["p_test"]
    low = int(np.argmin(p))
    assert RC.recourse(s, "xgb", 0.5, test_index=low, actionable=RULES)["status"] == "already_accepted"
    hard = RC.recourse(s, "xgb", 0.001, test_index=_refused_client(s), actionable=RULES)
    assert hard["status"] == "none_found" and hard["options"] == []
    assert hard["lowest_probability_found"] >= 0.001                           # on dit jusqu'où on est descendu
    s.name = "autre"
    assert RC.recourse(s, "xgb", THR, test_index=0)["status"] == "no_actionable"   # rien n'est déclaré : rien n'est inventé


def test_two_changes_never_do_worse_than_one_and_results_are_stable():
    s = _trained("xgb")
    i = _refused_client(s)
    one = RC.recourse(s, "xgb", THR, test_index=i, actionable=RULES, max_changes=1)
    two = RC.recourse(s, "xgb", THR, test_index=i, actionable=RULES, max_changes=2)
    assert all(len(o["changes"]) == 1 for o in one["options"])
    if one["options"] and two["options"]:
        assert two["options"][0]["score"] <= one["options"][0]["score"] + 1e-9
    assert two == RC.recourse(s, "xgb", THR, test_index=i, actionable=RULES, max_changes=2)


def test_text_contains_the_real_numbers_and_the_caveat():
    s = _trained("xgb")
    r = RC.recourse(s, "xgb", THR, test_index=_refused_client(s), actionable=RULES)
    o = r["options"][0]
    assert RC._pct(r["probability"], THR) in o["text"] and RC._pct(o["new_probability"], THR) in o["text"]
    assert "ne garantit" in r["caveat"]


def test_taiwan_configuration_is_safe():
    """Taïwan : rien ne change « maintenant » (tout est fixe ou passé) ; seul le comportement à venir est conseillé."""
    assert "taiwan_credit" not in RC.ACTIONABLE
    w = RC.TIME_WINDOWS["taiwan_credit"]
    assert w["months"] == 6 and {w["code_full"], w["code_partial"], w["code_none"]} == {-1, 0, -2}
    assert not any(X.is_sensitive(w[k].format(k=1)) for k in ("status", "bill", "paid"))


def test_recourse_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-recourse"}
    buf = io.StringIO()
    _credit().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("credit.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["age", "sex", "debt", "paid", "job"]}, headers=H).status_code == 200
    assert c.post("/api/models/train", json={"models": [{"name": "xgb"}]}, headers=H).status_code == 200
    body = {"model": "xgb", "threshold": 0.25, "test_index": 0, "actionable": {"paid": "up"}}
    r = c.post("/api/decision/recourse", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("found", "none_found", "already_accepted")
    assert c.post("/api/decision/recourse", json={**body, "actionable": {"age": "up"}}, headers=H).status_code == 400
    assert c.post("/api/decision/recourse", json={**body, "test_index": None}, headers=H).status_code == 400
    assert c.post("/api/decision/recourse", json={**body, "model": "rf"}, headers=H).status_code == 404
    assert c.post("/api/decision/recourse", json={"model": "xgb", "test_index": 0}, headers=H).json()["status"] == "no_actionable"


def test_values_are_rounded_like_a_human_would_say_them():
    s = _trained("xgb")
    r = RC.recourse(s, "xgb", THR, test_index=_refused_client(s), actionable=RULES)
    for o in r["options"]:
        for c in o["changes"]:
            assert float(f"{c['to']:.3g}") == c["to"] or abs(c["to"] - round(c["to"], 2)) < 1e-9


def test_no_two_change_option_contains_a_change_that_is_enough_alone():
    s = _trained("xgb")
    i = _refused_client(s)
    raw = X.client_matrices(s, test_index=i)[2]
    r = RC.recourse(s, "xgb", THR, test_index=i, actionable=RULES, n_options=5)
    est = s.models["xgb"]["est"]
    for o in r["options"]:
        if len(o["changes"]) == 2:
            for c in o["changes"]:
                _, Z, _, _ = X.client_matrices(s, record={**raw, c["feature"]: c["to"]})
                assert float(est.predict_proba(Z)[0, 1]) >= THR                # sinon le 2e changement est inutile


def test_percentages_never_look_equal_to_the_threshold():
    assert RC._pct(0.1697, 0.17) == "16.97%"                # 17.0 % sous 17.0 % serait incompréhensible
    assert RC._pct(0.151, 0.17) == "15.1%"
