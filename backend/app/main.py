"""Module 6 – API d'interprétation (FastAPI). Documentation interactive : /docs"""
import os
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import calibration as CAL
from . import data as D
from . import decision as DEC
from . import explain as X
from . import final as FIN
from . import metrics as MT
from . import models as MD
from . import prep as P
from . import reasons as RS
from . import recourse as RC
from . import recourse_audit as RA
from . import fraud as FR
from . import validation as V
from .state import Session, clean, get_session


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        import json
        return json.dumps(clean(content), allow_nan=False, ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="API d'interprétation – Classification explicable",
    description="Chargement de données, préparation, modélisation, évaluation et explications SHAP/LIME. "
                "Chaque session est isolée par l'en-tête `X-Session-Id`.",
    version="1.0.0", default_response_class=SafeJSON)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def sess_dep(x_session_id: str = Header(default="default")) -> Session:
    return get_session(x_session_id)


def need_data(s: Session):
    if s.df is None:
        raise HTTPException(400, "Aucun jeu de données chargé.")
    return s


def need_prep(s: Session):
    if s.Z_train is None:
        raise HTTPException(400, "Préparez d'abord les données (module 2).")
    return s


def need_model(s: Session, name: str):
    need_prep(s)
    if name not in s.models:
        raise HTTPException(404, f"Modèle « {name} » non entraîné.")
    return s.models[name]


def guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ santé
@app.get("/api/health", tags=["système"])
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ module 1
def _install(s: Session, df, target, positive, name):
    s.df, s.target, s.name = df.reset_index(drop=True), target, name
    s.positive = str(positive if positive is not None else D.default_positive(df[target]))
    s.Z_train = None
    s.models, s.cache = {}, {}


@app.get("/api/datasets", tags=["données"])
def datasets():
    return [{"key": k, "label": v["label"], "source": v["source"]} for k, v in D.PRESETS.items()]


class LoadReq(BaseModel):
    dataset: str = "telco"


@app.post("/api/data/load", tags=["données"])
def load(req: LoadReq, s: Session = Depends(sess_dep)):
    def go():
        try:
            df, target, positive = D.load_preset(req.dataset)
        except ValueError:
            raise
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(502, f"Téléchargement impossible : {e}")
        _install(s, df, target, positive, req.dataset)
        return D.summary(s.df, s.target, s.positive, req.dataset)
    return guard(go)


@app.post("/api/data/upload", tags=["données"])
async def upload(file: UploadFile = File(...), target: str = Form(None), s: Session = Depends(sess_dep)):
    content = await file.read()
    try:
        df, tgt = D.load_csv(content, target or None)
    except Exception as e:
        raise HTTPException(400, f"CSV illisible : {e}")
    if len(df) < 100:
        raise HTTPException(400, "Le fichier doit contenir au moins 100 observations.")
    _install(s, df, tgt, None, file.filename or "upload")
    return D.summary(s.df, s.target, s.positive, file.filename or "upload")


@app.get("/api/data/summary", tags=["données"])
def data_summary(s: Session = Depends(sess_dep)):
    need_data(s)
    return D.summary(s.df, s.target, s.positive, s.name)


class TargetReq(BaseModel):
    target: str
    positive: str | None = None


@app.post("/api/data/target", tags=["données"])
def set_target(req: TargetReq, s: Session = Depends(sess_dep)):
    need_data(s)
    if req.target not in s.df.columns:
        raise HTTPException(400, "Colonne cible inconnue.")
    s.target = req.target
    s.positive = str(req.positive) if req.positive is not None else D.default_positive(s.df[req.target])
    if s.positive not in set(s.df[req.target].dropna().astype(str)):
        raise HTTPException(400, "Cette modalité n'existe pas dans la cible.")
    s.Z_train, s.models, s.cache = None, {}, {}
    return D.summary(s.df, s.target, s.positive, s.name)


@app.get("/api/data/column/{col}", tags=["données"])
def data_column(col: str, s: Session = Depends(sess_dep)):
    need_data(s)
    if col not in s.df.columns:
        raise HTTPException(404, "Colonne inconnue.")
    return D.column_view(s.df, col, s.target, s.positive)


@app.get("/api/data/correlation", tags=["données"])
def data_corr(s: Session = Depends(sess_dep)):
    need_data(s)
    return D.correlation(s.df, s.target, s.positive)


# ------------------------------------------------------------------ module 2
class PrepReq(BaseModel):
    features: list[str]
    num_impute: str = "median"       # mean | median | zero | drop
    cat_impute: str = "mode"         # mode | drop
    scaling: str = "standard"        # standard | minmax | none
    test_size: float = Field(0.25, gt=0.05, lt=0.6)
    random_state: int = 42
    stratify: bool = True
    time_column: str | None = Field(None, description="colonne de temps : découpage temporel (test = les plus récents)")


@app.post("/api/prep", tags=["préparation"])
def prep(req: PrepReq, s: Session = Depends(sess_dep)):
    need_data(s)
    return guard(lambda: P.build(s, req.model_dump()))


# ------------------------------------------------------------------ module 3
@app.get("/api/models/catalog", tags=["modèles"])
def catalog():
    return {k: {"label": v["label"], "params": v["params"], "grid": v["grid"]} for k, v in MD.CATALOG.items()}


class ModelSpec(BaseModel):
    name: str
    params: dict = {}
    tune: bool = False


class TrainReq(BaseModel):
    models: list[ModelSpec]
    balance: bool = False
    threshold: float = 0.5


@app.post("/api/models/train", tags=["modèles"])
def train(req: TrainReq, s: Session = Depends(sess_dep)):
    need_prep(s)
    out = []
    for spec in req.models:
        if spec.name not in MD.CATALOG:
            raise HTTPException(400, f"Modèle inconnu : {spec.name}")
        est, info = MD.train_one(spec.name, spec.params, spec.tune, req.balance, s.Z_train, s.y_train.values)
        p = est.predict_proba(s.Z_test)[:, 1]
        s.models[spec.name] = {"est": est, "info": info, "p_test": p, "label": MD.CATALOG[spec.name]["label"]}
        s.cache = {k: v for k, v in s.cache.items() if not (isinstance(k, tuple) and k[0] == spec.name)}
        out.append({"name": spec.name, "label": MD.CATALOG[spec.name]["label"], **info})
    s.cache.pop("lime", None)
    return {"trained": out}


class CompareReq(BaseModel):
    models: list[ModelSpec]
    balance: bool = False
    n_splits: int = Field(5, ge=3, le=10)
    n_repeats: int = Field(2, ge=1, le=5)
    max_rows: int = Field(20000, ge=1000, le=100000)


@app.post("/api/models/compare", tags=["modèles"])
def compare(req: CompareReq, s: Session = Depends(sess_dep)):
    """Compare les modèles par validation croisée répétée sur le lot d'apprentissage SEUL (jamais le test)."""
    need_prep(s)
    for spec in req.models:
        if spec.name not in MD.CATALOG:
            raise HTTPException(400, f"Modèle inconnu : {spec.name}")
    specs = [{"name": m.name, "params": m.params} for m in req.models]
    return guard(lambda: V.cross_validate(s, specs, req.n_splits, req.n_repeats, balance=req.balance,
                                          max_rows=req.max_rows))


class CalibReq(BaseModel):
    model: ModelSpec
    balance: bool = False
    n_splits: int = Field(5, ge=3, le=10)
    max_rows: int = Field(20000, ge=1000, le=100000)


@app.post("/api/models/calibration", tags=["modèles"])
def calibration(req: CalibReq, s: Session = Depends(sess_dep)):
    """Mesure la fiabilité des probabilités (Brier, courbe de fiabilité) sur le lot d'apprentissage SEUL."""
    need_prep(s)
    if req.model.name not in MD.CATALOG:
        raise HTTPException(400, f"Modèle inconnu : {req.model.name}")
    spec = {"name": req.model.name, "params": req.model.params}
    return guard(lambda: CAL.calibration_report(s, spec, req.n_splits, balance=req.balance, max_rows=req.max_rows))


class ThresholdReq(BaseModel):
    model: ModelSpec
    cost_fn: float = Field(5.0, gt=0, description="coût d'un cas manqué (crédit accordé à un client qui fait défaut)")
    cost_fp: float = Field(1.0, gt=0, description="coût d'une fausse alerte (client sain refusé)")
    balance: bool = False
    n_splits: int = Field(5, ge=3, le=10)
    max_rows: int = Field(20000, ge=1000, le=100000)


@app.post("/api/models/threshold", tags=["modèles"])
def threshold(req: ThresholdReq, s: Session = Depends(sess_dep)):
    """Seuil de décision qui minimise le coût des erreurs, sur le lot d'apprentissage SEUL (jamais le test)."""
    need_prep(s)
    if req.model.name not in MD.CATALOG:
        raise HTTPException(400, f"Modèle inconnu : {req.model.name}")
    spec = {"name": req.model.name, "params": req.model.params}
    return guard(lambda: DEC.decision_report(s, spec, req.cost_fn, req.cost_fp, req.n_splits,
                                             balance=req.balance, max_rows=req.max_rows))


class FinalReq(ThresholdReq):
    force: bool = False


@app.post("/api/final/evaluate", tags=["évaluation finale"])
def final_evaluate(req: FinalReq, s: Session = Depends(sess_dep)):
    """Ouvre le lot de test UNE fois pour juger la chaîne complète (modèle, calibration, seuil).
    Un second appel renvoie le résultat déjà obtenu ; `force` rouvre le test et le compte."""
    need_prep(s)
    if req.model.name not in MD.CATALOG:
        raise HTTPException(400, f"Modèle inconnu : {req.model.name}")
    spec = {"name": req.model.name, "params": req.model.params}
    return guard(lambda: FIN.final_evaluation(s, spec, req.cost_fn, req.cost_fp, req.n_splits,
                                              balance=req.balance, max_rows=req.max_rows, force=req.force))


# ------------------------------------------------------------------ module 4
@app.get("/api/evaluate", tags=["évaluation"])
def evaluate(threshold: float = 0.5, s: Session = Depends(sess_dep)):
    need_prep(s)
    if not s.models:
        raise HTTPException(400, "Entraînez au moins un modèle.")
    res = {}
    for name, m in s.models.items():
        ev = MT.evaluate(s.y_test.values, m["p_test"], threshold)
        ev.update({"label": m["label"], "params": m["info"].get("params"), "tuned": m["info"].get("tuned"),
                   "train_seconds": m["info"].get("train_seconds"), "cv_auc": m["info"].get("cv_auc"),
                   "best_f1_threshold": MT.best_f1_threshold(s.y_test.values, m["p_test"])})
        res[name] = ev
    return {"threshold": threshold, "positive_rate": float(s.y_test.mean()), "n_test": int(len(s.y_test)),
            "imbalanced": bool(s.y_test.mean() < 0.3 or s.y_test.mean() > 0.7), "models": res}


# ------------------------------------------------------------------ module 5
@app.get("/api/explain/global/{name}", tags=["explications"])
def explain_global(name: str, s: Session = Depends(sess_dep)):
    need_model(s, name)
    return {"native": X.native_importance(s, name), "shap": X.shap_global(s, name)}


@app.get("/api/explain/permutation", tags=["explications"])
def explain_perm(models: str = "", s: Session = Depends(sess_dep)):
    need_prep(s)
    names = [m for m in models.split(",") if m] or list(s.models)
    for n in names:
        need_model(s, n)
    return {n: X.permutation(s, n) for n in names}


@app.get("/api/explain/form", tags=["explications"])
def explain_form(s: Session = Depends(sess_dep)):
    need_prep(s)
    return {"fields": P.form_schema(s), "n_test": int(len(s.y_test))}


@app.get("/api/explain/test-samples/{name}", tags=["explications"])
def test_samples(name: str, threshold: float = 0.5, limit: int = 40, s: Session = Depends(sess_dep)):
    m = need_model(s, name)
    p = m["p_test"]
    order = np.argsort(-p)[:limit]
    return [{"index": int(i), "proba": float(p[i]), "true": int(s.y_test.iloc[i])} for i in order]


class LocalReq(BaseModel):
    model: str
    threshold: float = 0.5
    test_index: int | None = None
    record: dict | None = None


@app.post("/api/explain/local", tags=["explications"])
def explain_local(req: LocalReq, s: Session = Depends(sess_dep)):
    need_model(s, req.model)
    if req.test_index is None and req.record is None:
        raise HTTPException(400, "Fournissez test_index ou record.")
    return guard(lambda: X.local(s, req.model, req.threshold, req.test_index, req.record))


@app.get("/api/explain/sensitive/{name}", tags=["explications"])
def explain_sensitive(name: str, threshold: float = 0.5, s: Session = Depends(sess_dep)):
    need_model(s, name)
    return X.sensitive_audit(s, name, threshold)


class ReasonsReq(BaseModel):
    model: str
    threshold: float = Field(0.5, gt=0, lt=1)
    test_index: int | None = None
    record: dict | None = None
    top: int = Field(3, ge=1, le=10)


@app.post("/api/decision/reasons", tags=["décision"])
def decision_reasons(req: ReasonsReq, s: Session = Depends(sess_dep)):
    """Décision (accord/refus) d'un client et ses motifs principaux, sans variables sensibles."""
    need_model(s, req.model)
    if req.test_index is None and req.record is None:
        raise HTTPException(400, "Fournissez test_index ou record.")
    return guard(lambda: RS.decision_reasons(s, req.model, req.threshold, req.test_index, req.record, req.top))


class RecourseReq(BaseModel):
    model: str
    threshold: float = Field(0.5, gt=0, lt=1)
    test_index: int | None = None
    record: dict | None = None
    actionable: dict[str, str] | None = Field(None, description="variable -> sens autorisé (up, down, both)")
    max_changes: int = Field(2, ge=1, le=2)
    n_options: int = Field(3, ge=1, le=5)


@app.post("/api/decision/recourse", tags=["décision"])
def decision_recourse(req: RecourseReq, s: Session = Depends(sess_dep)):
    """Conseils réalistes pour passer de « refus » à « accord » : variables modifiables seulement, vérifiés par le modèle."""
    need_model(s, req.model)
    if req.test_index is None and req.record is None:
        raise HTTPException(400, "Fournissez test_index ou record.")
    return guard(lambda: RC.recourse(s, req.model, req.threshold, req.test_index, req.record, req.actionable,
                                     req.max_changes, req.n_options))


class AuditReq(BaseModel):
    model: str
    threshold: float = Field(0.5, gt=0, lt=1)
    max_clients: int | None = Field(400, ge=10, le=20000, description="null = tous les clients refusés")
    actionable: dict[str, str] | None = None
    max_changes: int = Field(2, ge=1, le=2)


@app.post("/api/decision/audit", tags=["décision"])
def decision_audit(req: AuditReq, s: Session = Depends(sess_dep)):
    """Audit du recours sur l'ensemble des clients refusés : part avec conseil, effort demandé, équité par groupe."""
    need_model(s, req.model)
    return guard(lambda: RA.audit(s, req.model, req.threshold, req.max_clients, 0, req.actionable, req.max_changes))


# ------------------------------------------------------------------ fraude (phase C)
class FraudReq(BaseModel):
    review_cost: float = Field(5.0, gt=0, description="coût d'une vérification (même unité que le montant)")
    amount_column: str = "Amount"
    n_splits: int = Field(4, ge=2, le=8)
    variants: list[tuple[str, bool]] | None = Field(None, description="(modèle, rééquilibrer ?) ; défaut : logreg et xgb, avec et sans")


class FraudFinalReq(FraudReq):
    force: bool = False


@app.post("/api/fraud/report", tags=["fraude"])
def fraud_report(req: FraudReq, s: Session = Depends(sess_dep)):
    """Compare modèles et stratégies d'alerte sur l'apprentissage seul (validation glissante dans le temps)."""
    need_prep(s)
    return guard(lambda: FR.fraud_report(s, req.variants, req.review_cost, req.n_splits, req.amount_column))


@app.post("/api/fraud/final", tags=["fraude"])
def fraud_final(req: FraudFinalReq, s: Session = Depends(sess_dep)):
    """Décide sur l'apprentissage, puis ouvre le lot de test une seule fois (ouverture comptée)."""
    need_prep(s)
    return guard(lambda: FR.fraud_final(s, req.review_cost, req.variants, req.n_splits, req.amount_column, req.force))


# ------------------------------------------------------------------ interface web (build React)
_DIST = Path(os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="web")
