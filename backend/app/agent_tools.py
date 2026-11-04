"""Les outils de l'agent (phase D) : la SEULE source de chiffres et de faits qu'il a le droit d'utiliser.

Chaque outil appelle les modules déjà testés (reasons, recourse, recourse_audit, fraud, final) et renvoie un petit
dictionnaire JSON, en français, avec des pourcentages déjà calculés (un petit LLM se trompe facilement en convertissant
0,348 en 34,8 %). Un outil ne lève jamais d'exception vers l'agent : en cas de problème, il renvoie {"erreur": ...},
que le LLM peut lire et transmettre honnêtement.
"""
from dataclasses import dataclass, field

import numpy as np

from . import models as MD
from . import reasons as R
from . import recourse as RC
from . import recourse_audit as RA
from .state import Session


def _pct(x, d: int = 1):
    return None if x is None else round(100 * float(x), d)


@dataclass
class ToolContext:
    """Ce que les outils ont besoin de savoir : la session, le modèle expliqué, le seuil de décision."""
    sess: Session
    model: str
    threshold: float
    cache: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MD.CATALOG[self.model]["label"] if self.model in MD.CATALOG else self.model


def _client(ctx: ToolContext, numero) -> int:
    n = len(ctx.sess.y_test)
    try:
        i = int(numero)
    except (TypeError, ValueError):
        raise ValueError(f"Numéro de client invalide : {numero!r} (entier entre 0 et {n - 1}).")
    if not 0 <= i < n:
        raise ValueError(f"Le client n°{i} n'existe pas : les numéros vont de 0 à {n - 1}.")
    return i


def decision_client(ctx: ToolContext, numero) -> dict:
    i = _client(ctx, numero)
    est = ctx.sess.models[ctx.model]["est"]
    p = float(est.predict_proba(ctx.sess.Z_test[[i]])[0, 1])
    if ctx.sess.name == "ulb_fraud":                          # fraude : la décision dépend du montant
        A = float(ctx.sess.df.loc[ctx.sess.X_test_raw.index[i], "Amount"])
        c = float((ctx.sess.fraud_choice or {}).get("review_cost", 5.0))
        return {"numero": i, "modele": ctx.label, "probabilite_fraude_pct": _pct(p, 2), "montant": round(A, 2),
                "cout_verification": c, "perte_moyenne_si_non_verifiee": round(p * A, 2),
                "a_verifier": bool(p * A > c),
                "regle": "vérifier si probabilité × montant > coût d'une vérification"}
    return {"numero": i, "modele": ctx.label, "risque_annonce_pct": _pct(p), "seuil_pct": _pct(ctx.threshold),
            "decision": "refus" if p >= ctx.threshold else "accord",
            "ecart_au_seuil_points": _pct(p - ctx.threshold)}


def motifs_client(ctx: ToolContext, numero) -> dict:
    i = _client(ctx, numero)
    r = R.decision_reasons(ctx.sess, ctx.model, ctx.threshold, test_index=i)
    out = {"numero": i, "decision": r["decision"], "risque_annonce_pct": _pct(r["probability"]),
           "motifs_vers_le_refus": [{"texte": x["text"], "part_pct": _pct(x["share_of_push"], 0)}
                                    for x in r["reasons_against"]],
           "elements_en_faveur": [{"texte": x["text"], "part_pct": _pct(x["share_of_push"], 0)}
                                  for x in r["reasons_for"]],
           "variables_sensibles_jamais_citees": r["excluded_sensitive"],
           "rappel": "Ces motifs décrivent ce que le modèle a appris (des corrélations), pas des causes."}
    if r.get("note"):
        out["mise_en_garde"] = r["note"]
    return out


def conseil_client(ctx: ToolContext, numero) -> dict:
    i = _client(ctx, numero)
    r = RC.recourse(ctx.sess, ctx.model, ctx.threshold, test_index=i)
    out = {"numero": i, "statut": r["status"], "risque_annonce_pct": _pct(r["probability"]),
           "seuil_pct": _pct(ctx.threshold), "avertissement": r.get("caveat")}
    if r["status"] == "found":
        out["conseils"] = [{"texte": o["text"], "delai_mois": o.get("horizon_months"),
                            "montant_mensuel": o.get("monthly_payment"),
                            "nouveau_risque_pct": _pct(o["new_probability"])} for o in r["options"]]
    if r["status"] == "none_found":
        out["risque_le_plus_bas_atteignable_pct"] = _pct(r["lowest_probability_found"])
    if r.get("assumptions"):
        out["hypotheses"] = r["assumptions"]
    if r.get("reason"):
        out["raison"] = r["reason"]
    out["signification_du_statut"] = {
        "found": "des conseils réalistes existent (vérifiés par le modèle)",
        "none_found": "aucun conseil réaliste : même au mieux, le risque reste au-dessus du seuil",
        "already_accepted": "le client est déjà accepté au seuil choisi",
        "no_actionable": "aucune variable modifiable déclarée pour ce jeu : pas de conseil plutôt qu'un conseil inventé",
        "not_applicable": "pas de conseil pour ce type de décision",
    }.get(r["status"], r["status"])
    return out


def audit_recours(ctx: ToolContext) -> dict:
    key = ("audit", ctx.model, id(ctx.sess.models[ctx.model]["est"]), round(ctx.threshold, 4))   # réentraîner = recalculer
    if key not in ctx.cache:
        ctx.cache[key] = RA.audit(ctx.sess, ctx.model, ctx.threshold, max_clients=400)
    a = ctx.cache[key]
    out = {"statut": a["status"], "seuil_pct": _pct(ctx.threshold), "clients_refuses": a.get("n_refused"),
           "clients_du_test": a.get("n_test")}
    if a["status"] != "ok":
        out["raison"] = a.get("reason")
        return out
    out.update({"clients_audites": a["n_audited"], "echantillon_au_hasard": a["sampled"],
                "part_avec_conseil_pct": _pct(a["share_found"]),
                "intervalle_95_pct": [_pct(v) for v in a["share_found_ci95"]] if a["sampled"] else None,
                "part_sans_conseil_pct": _pct(a["share_none"]),
                "equite_acces_au_conseil": [
                    {"variable": f["variable"], "ecart_points": _pct(f["gap"]), "alerte": f["alert"],
                     "groupes": [{"groupe": g["group"], "part_avec_conseil_pct": _pct(g["share_found"]),
                                  "effectif": g["n"]} for g in f["groups"]]} for f in a["fairness"]]})
    if a.get("none_found", {}).get("share_no_bill") is not None:
        out["parmi_les_sans_conseil_part_sans_facture_pct"] = _pct(a["none_found"]["share_no_bill"])
    return out


def infos_modele(ctx: ToolContext) -> dict:
    s = ctx.sess
    out = {"jeu_de_donnees": s.name, "cible": f"{s.target} = {s.positive}", "modele_explique": ctx.label,
           "seuil_pct": _pct(ctx.threshold), "clients_apprentissage": int(len(s.y_train)),
           "clients_test": int(len(s.y_test)), "taux_positif_apprentissage_pct": _pct(float(np.mean(s.y_train)), 2),
           "modeles_entraines": [MD.CATALOG[m]["label"] for m in s.models if m in MD.CATALOG],
           "decoupage": "dans le temps" if s.prep_cfg.get("time_column") else "au hasard",
           "ouvertures_du_lot_de_test": s.test_openings}
    f = s.final
    if f:
        out["evaluation_finale"] = {
            "modele": f["label"], "seuil_retenu": round(f["threshold"], 3),
            "couts_hypotheses": f"un défaut manqué coûte {f['costs']['cost_fn']:g}, une fausse alerte {f['costs']['cost_fp']:g}",
            "cout_promis_par_dossier": round(f["expected"]["cost"], 3),
            "cout_reel_par_dossier": round(f["test"]["at_threshold"]["cost"], 3),
            "intervalle_95": [round(v, 3) for v in f["test"]["cost_ci95"]],
            "auc_test": round(f["test"]["auc"], 3)}
    else:
        out["evaluation_finale"] = "pas encore faite (carte « Évaluation finale » de l'onglet Modèles)"
    return out


def fraude_resume(ctx: ToolContext) -> dict:
    s = ctx.sess
    if s.fraud_final:
        f = s.fraud_final
        t = f["test"]
        return {"statut": "test ouvert", "choix": f"{f['choice']['label']} + {f['choice']['strategy_label']}",
                "cout_verification": f["review_cost"],
                "pertes_evitees_promises_pct": _pct(f["expected"]["saved_share"]),
                "pertes_evitees_test_pct": _pct(t["chosen"]["saved_share"]),
                "fraudes_attrapees_test_pct": _pct(t["chosen"]["recall"]),
                "alertes_justifiees_test_pct": _pct(t["chosen"]["precision"]),
                "alertes_pour_1000": round(t["chosen"]["alerts_per_1000"], 2),
                "fraudes_dans_le_test": t["chosen"]["frauds_total"], "pr_auc_test": round(t["pr_auc"], 3),
                "exactitude_si_rien_signale_pct": _pct(t["accuracy_accept_all"], 2),
                "taux_de_fraude_apprentissage_pct": _pct(f["train_base_rate"], 2),
                "taux_de_fraude_test_pct": _pct(t["base_rate"], 2)}
    if s.fraud_choice:
        e = s.fraud_choice["expected"]
        return {"statut": "choix fait sur l'apprentissage, test pas encore ouvert",
                "choix": f"{s.fraud_choice['label']} + {s.fraud_choice['strategy_label']}",
                "pertes_evitees_promises_pct": _pct(e["saved_share"]), "fraudes_attrapees_pct": _pct(e["recall"])}
    return {"statut": "pas encore calculé : utilisez la carte « Fraude » de l'onglet Modèles"}


GLOSSARY = {
    "seuil": "Probabilité à partir de laquelle on refuse. Choisi par le coût des erreurs : seuil = coût d'une fausse "
             "alerte / (coût d'une fausse alerte + coût d'un défaut manqué). Avec des coûts 1 et 5 : 1/6, soit 16,7 %.",
    "calibration": "Une probabilité est calibrée si, parmi les clients annoncés à 20 %, environ 20 % font vraiment "
                   "défaut. On la mesure avec le score de Brier et la courbe de fiabilité.",
    "brier": "Score de Brier : moyenne de (probabilité annoncée − ce qui s'est passé)². Plus petit = mieux ; annoncer "
             "toujours 50 % donne 0,25.",
    "auc": "AUC (aire sous la courbe ROC) : probabilité que le modèle classe un cas positif au-dessus d'un négatif. "
           "0,5 = hasard, 1 = parfait. Elle mesure l'ordre, pas la justesse des probabilités.",
    "pr-auc": "PR-AUC (aire sous la courbe précision-rappel) : résume précision et rappel sur tous les seuils. Utile "
              "quand les cas positifs sont rares (fraude).",
    "shap": "SHAP (SHapley Additive exPlanations) : part de chaque variable dans la prédiction d'un client, par rapport "
            "à un client moyen. Les parts s'additionnent jusqu'à la prédiction.",
    "motif": "Motif de refus : une des variables qui poussent le plus vers le refus, d'après SHAP. L'âge, le sexe et la "
             "situation maritale ne sont jamais cités comme motifs.",
    "recours": "Conseil réaliste qui ferait passer le risque sous le seuil : seulement ce que le client contrôle, pour "
               "l'avenir (le passé ne se réécrit pas), vérifié par le modèle. Ce n'est pas une garantie.",
    "audit": "Audit du recours : on cherche un conseil pour chaque client refusé et on compte, y compris par groupe "
             "(sexe, âge), pour vérifier que l'accès au conseil est équitable.",
    "validation croisée": "On découpe l'apprentissage en plis : on apprend sur les uns, on note sur l'autre, et on "
                          "tourne. Sert à choisir un modèle sans toucher au lot de test.",
    "lot de test": "Données mises de côté au début, jamais utilisées pour choisir. On ne l'ouvre qu'une fois, pour "
                   "vérifier la promesse.",
    "rappel": "Rappel : part des cas positifs (défauts, fraudes) que le modèle attrape.",
    "précision": "Précision : part des alertes qui sont de vrais cas positifs.",
    "règle du montant": "Fraude : vérifier une transaction si probabilité × montant > coût d'une vérification.",
}


def definir(ctx: ToolContext, terme) -> dict:
    t = str(terme or "").strip().lower()
    for k, v in GLOSSARY.items():
        if t == k or t in k or k in t:
            return {"terme": k, "definition": v}
    return {"erreur": f"terme inconnu : {terme!r}", "termes_connus": sorted(GLOSSARY)}


_NUM = {"type": "object", "properties": {"numero": {"type": "integer", "description": "numéro du client dans le lot de test (0, 1, 2…)"}},
        "required": ["numero"]}
_NONE = {"type": "object", "properties": {}}
TOOLS = {
    "decision_client": (decision_client, "Décision pour un client du lot de test : risque annoncé, seuil, refus ou accord (fraude : alerte ou non).", _NUM),
    "motifs_client": (motifs_client, "Motifs principaux de la décision d'un client (SHAP), sans variable sensible.", _NUM),
    "conseil_client": (conseil_client, "Ce que le client pourrait faire pour être accepté (conseils vérifiés), ou pourquoi il n'y en a pas.", _NUM),
    "audit_recours": (audit_recours, "Sur tous les clients refusés : part avec un conseil, et équité de l'accès au conseil par groupe (sexe, âge…).", _NONE),
    "infos_modele": (infos_modele, "Jeu de données, modèle expliqué, seuil, découpage, évaluation finale (coût promis et réel).", _NONE),
    "fraude_resume": (fraude_resume, "Résultats de la détection de fraude : stratégie choisie, pertes évitées, fraudes attrapées.", _NONE),
    "definir": (definir, "Définition simple d'un terme du projet (seuil, calibration, SHAP, recours, PR-AUC…).",
                {"type": "object", "properties": {"terme": {"type": "string"}}, "required": ["terme"]}),
}


def schemas() -> list[dict]:
    """Description des outils au format attendu par Ollama (et par la plupart des API de LLM)."""
    return [{"type": "function", "function": {"name": k, "description": d, "parameters": p}}
            for k, (_, d, p) in TOOLS.items()]


def run_tool(ctx: ToolContext, name: str, arguments: dict | None) -> dict:
    """Exécute un outil ; toute erreur devient un message lisible par le LLM (jamais une exception)."""
    if name not in TOOLS:
        return {"erreur": f"outil inconnu : {name}", "outils_disponibles": sorted(TOOLS)}
    fn, _, params = TOOLS[name]
    args = dict(arguments or {})
    allowed = set(params.get("properties", {}))
    try:
        return fn(ctx, **{k: v for k, v in args.items() if k in allowed})
    except TypeError:
        return {"erreur": f"arguments attendus pour {name} : {sorted(params.get('required', []))}"}
    except Exception as e:                                      # noqa: BLE001 - l'agent doit pouvoir le lire
        return {"erreur": str(e)}
