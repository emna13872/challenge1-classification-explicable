"""Évaluation finale : le test s'ouvre une fois, les ouvertures sont comptées, et la promesse est comparée au réel.

Données synthétiques fabriquées ici : ces tests tournent partout (CI comprise), sans fichier externe.
"""
import io

from fastapi.testclient import TestClient

from app import final as F
from app import prep as P
from app.main import app
from test_validation import _session, _synthetic

SPEC = {"name": "logreg"}


def test_first_call_opens_the_test_once_and_second_call_does_not():
    s = _session()
    a = F.final_evaluation(s, SPEC, cost_fn=5, cost_fp=1, n_splits=3)
    assert a["openings"] == 1 and a["already_opened"] is False and a["warning"] is None
    b = F.final_evaluation(s, SPEC, cost_fn=5, cost_fp=1, n_splits=3)
    assert b["already_opened"] is True and s.test_openings == 1          # pas de nouvelle lecture du test
    assert b["test"]["auc"] == a["test"]["auc"]


def test_force_reopens_counts_and_warns():
    s = _session()
    F.final_evaluation(s, SPEC, 5, 1, n_splits=3)
    c = F.final_evaluation(s, SPEC, 2, 1, n_splits=3, force=True)
    assert c["openings"] == 2 and "2 fois" in c["warning"]


def test_new_split_resets_the_counter():
    s = _session()
    F.final_evaluation(s, SPEC, 5, 1, n_splits=3)
    P.build(s, {"features": ["x1", "x2", "cat"], "test_size": 0.25, "random_state": 7, "num_impute": "median",
                "cat_impute": "mode", "scaling": "standard", "stratify": True})
    assert s.test_openings == 0 and s.final is None                      # nouveau lot de test, de nouveau vierge


def test_decisions_are_made_without_the_test_set():
    """Les trois décisions (probabilités, correction, seuil) viennent de l'apprentissage : on le prouve en
    changeant les étiquettes du test entre deux évaluations ; seuil et méthode ne doivent pas bouger."""
    a = _session()
    b = _session()
    b.y_test = 1 - b.y_test                                              # étiquettes de test sabotées
    ra = F.final_evaluation(a, SPEC, 5, 1, n_splits=3)
    rb = F.final_evaluation(b, SPEC, 5, 1, n_splits=3)
    assert ra["threshold"] == rb["threshold"] and ra["calibration_used"] == rb["calibration_used"]
    assert ra["expected"]["cost"] == rb["expected"]["cost"]
    assert ra["test"]["at_threshold"]["cost"] != rb["test"]["at_threshold"]["cost"]


def test_promise_is_close_to_reality_and_calibration_helps_when_classes_are_balanced():
    s = _session()
    r = F.final_evaluation(s, SPEC, 5, 1, n_splits=5, balance=True)
    assert abs(r["test"]["at_threshold"]["cost"] - r["expected"]["cost"]) < 0.15
    assert r["test"]["brier_calibrated"] < r["test"]["brier_raw"]        # le rééquilibrage gonflait les probabilités
    lo, hi = r["test"]["cost_ci95"]
    assert lo <= r["test"]["at_threshold"]["cost"] <= hi


def test_final_endpoint_through_csv_upload():
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-final"}
    buf = io.StringIO()
    _synthetic().to_csv(buf, index=False)
    assert c.post("/api/data/upload", files={"file": ("synthetic.csv", buf.getvalue().encode(), "text/csv")},
                  data={"target": "y"}, headers=H).status_code == 200
    assert c.post("/api/prep", json={"features": ["x1", "x2", "cat"]}, headers=H).status_code == 200
    body = {"model": {"name": "logreg"}, "cost_fn": 5, "cost_fp": 1, "n_splits": 3}
    r1 = c.post("/api/final/evaluate", json=body, headers=H).json()
    r2 = c.post("/api/final/evaluate", json=body, headers=H).json()
    assert r1["openings"] == 1 and r2["already_opened"] is True and r2["openings"] == 1
    assert c.post("/api/final/evaluate", json={**body, "force": True}, headers=H).json()["openings"] == 2
    assert c.post("/api/final/evaluate", json={"model": {"name": "nope"}}, headers=H).status_code == 400


def test_final_requires_prepared_data():
    c = TestClient(app)
    r = c.post("/api/final/evaluate", json={"model": {"name": "logreg"}}, headers={"X-Session-Id": "vide-final"})
    assert r.status_code == 400
