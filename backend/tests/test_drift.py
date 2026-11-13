"""Suivi de dérive (phase F) : le PSI vaut ~0 quand rien ne change, grimpe quand on décale, et l'alerte se déclenche."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import drift as DR
from app.main import app
from test_agent import ctx  # noqa: F401  (session synthétique au format de Taïwan, XGBoost entraîné)


def test_psi_on_known_distributions():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert DR.psi(a, b) < 0.01                                   # même distribution : rien
    assert 0.10 < DR.psi(a, rng.normal(0.5, 1, 20000)) < 0.40    # décalage d'un demi écart-type : à surveiller
    assert DR.psi(a, rng.normal(1.5, 1, 20000)) > 1.0            # gros décalage : dérive nette
    assert DR.psi(a, a) == pytest.approx(0, abs=1e-12)


def test_psi_by_hand_on_two_categories():
    # référence 50/50, actuel 80/20 : (0,8−0,5)·ln(0,8/0,5) + (0,2−0,5)·ln(0,2/0,5)
    expected = 0.3 * np.log(1.6) + (-0.3) * np.log(0.4)
    got = DR.psi(["F"] * 50 + ["M"] * 50, ["F"] * 80 + ["M"] * 20, categorical=True)
    assert got == pytest.approx(expected, rel=1e-6)
    assert DR.psi(["F"] * 100, ["F"] * 90 + ["X"] * 10, categorical=True) > 0.25   # modalité jamais vue


def test_levels():
    assert [DR.level(x) for x in (0.05, 0.1, 0.2, 0.3)] == ["stable", "à surveiller", "à surveiller", "dérive"]


def test_test_set_is_stable_and_a_shift_raises_the_alert(ctx):  # noqa: F811
    s = ctx.sess
    base = DR.drift_report(s, "xgb", threshold=0.25)
    assert not base["alert"] and base["score"]["level"] == "stable"
    assert base["features"] == sorted(base["features"], key=lambda r: -r["psi"])
    moved = DR.drift_report(s, "xgb", DR.shifted(s, "PAY_1", 1, 2), 0.25, "décalé")   # deux mois de retard en plus
    top = moved["features"][0]
    assert moved["alert"] and top["feature"] == "PAY_1" and top["level"] == "dérive"
    assert moved["score"]["psi"] > base["score"]["psi"]
    assert moved["score"]["refusal_rate_current"] > moved["score"]["refusal_rate_reference"]
    assert s.X_test_raw["PAY_1"].max() < 10                     # le vrai lot de test n'a pas été modifié


def test_shift_input_errors(ctx):  # noqa: F811
    with pytest.raises(ValueError):
        DR.shifted(ctx.sess, "inconnue", 2)
    with pytest.raises(ValueError, match="numérique"):
        DR.shifted(ctx.sess, "SEX", 2)


def test_drift_route():
    from test_reasons import _credit
    c = TestClient(app)
    H = {"X-Session-Id": "pytest-drift"}
    assert c.post("/api/drift", json={}, headers=H).status_code == 400            # pas encore préparé
    buf = io.StringIO()
    _credit().to_csv(buf, index=False)
    c.post("/api/data/upload", files={"file": ("c.csv", buf.getvalue().encode(), "text/csv")}, data={"target": "y"}, headers=H)
    c.post("/api/prep", json={"features": ["age", "sex", "debt", "paid", "job"]}, headers=H)
    c.post("/api/models/train", json={"models": [{"name": "xgb"}]}, headers=H)
    r = c.post("/api/drift", json={"model": "xgb", "threshold": 0.3}, headers=H)
    assert r.status_code == 200 and "score" in r.json() and not r.json()["alert"]
    sh = c.post("/api/drift", json={"model": "xgb", "scenario": "shifted", "feature": "debt", "factor": 4}, headers=H)
    assert sh.status_code == 200 and sh.json()["alert"] and sh.json()["features"][0]["feature"] == "debt"
    assert c.post("/api/drift", json={"scenario": "shifted"}, headers=H).status_code == 400
