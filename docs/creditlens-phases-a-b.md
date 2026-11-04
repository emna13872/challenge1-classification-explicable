# CreditLens : moteur de risque et décision expliquée (phases A et B)

Branche `creditlens`. Jeu principal : *Default of Credit Card Clients* (UCI n° 350, Taïwan 2005), 30 000 clients, 22,1 % de défauts. Lot de test : 7 500 clients (25 %, stratifié, graine 42), ouvert une seule fois pour l'évaluation finale. Modèle de référence : XGBoost (réglages par défaut).

## Phase A : un risque fiable et une décision justifiée

| Étape | Module | Ce qui est fait | Résultat sur Taïwan |
|---|---|---|---|
| A1 Choix du modèle | `validation.py` | validation croisée stratifiée répétée sur l'apprentissage seul, préparation réapprise dans chaque pli ; prédictions hors-pli | RF 0,7819 ± 0,0048, XGBoost 0,7815 ± 0,0061 (égalité), arbre 0,7539, kNN 0,7445, régression logistique 0,7236 |
| A3 Calibration | `calibration.py` | Brier, ECE, courbe de fiabilité ; correction sigmoïde ou isotonique gardée seulement si elle améliore le Brier hors-pli | XGBoost déjà calibré (Brier 0,132) ; régression logistique rééquilibrée : 45,8 % annoncés pour 22,1 % réels, Brier 0,204 → 0,142 |
| A4 Seuil | `decision.py` | seuil = coût fausse alerte / (coût fausse alerte + coût défaut manqué) ; courbe coût-seuil | à 5:1 (hypothèse), seuil 0,167 : coût par dossier 0,547 contre 0,745 à 0,5 (−27 %) |
| A5 Évaluation finale | `final.py` | tout est choisi sur l'apprentissage ; test ouvert une fois (compteur `test_openings`) ; IC bootstrap à 95 % | XGBoost : coût promis 0,548, réel 0,560 (IC 0,536 à 0,585) |
| A6 Catalogue | `models.py` | une seule source de vérité pour chaque modèle (type, espace SHAP, libellés) | ajouter un modèle = 1 entrée + 1 ligne |

## Phase B : expliquer, conseiller, auditer

### B1 Motifs de refus : `reasons.py`, `POST /api/decision/reasons`
Contributions SHAP regroupées par variable d'origine (one-hot additionné), 3 motifs au plus avec la valeur du client et la médiane d'apprentissage. Les variables sensibles (âge, sexe, situation maritale) ne sont **jamais** des motifs ; elles sont listées à part et auditées. Contrôle d'additivité : écart maximal 1,4e-6 sur 150 refusés.

### B2 Recours : `recourse.py`, `POST /api/decision/recourse`
Toutes les variables de Taïwan sont fixes (limite fixée par la banque, données démographiques) ou **passées** (statuts, factures, montants remboursés sur 6 mois). Un conseil qui modifie le passé n'est pas réalisable. Le mode **« avenir »** répond donc à : *« si le client refait une demande dans m mois (m = 1, 3, 6), sans aucun retard, en remboursant au moins R par mois, quel serait son risque ? »*
- **Fenêtre glissante** : les m mois neufs entrent, les m plus anciens sortent ; l'historique restant et les variables fixes ne bougent pas (`_shifted`).
- **R** : montants observés (quantiles de `PAY_AMT1`), arrondis, au plus au 95e centile, jamais au-dessus de la facture (`payment_candidates`). On garde le plus petit R qui suffit pour chaque délai.
- **Vérification** : chaque scénario passe par la même préparation (`prep.row_to_matrices`) et le même modèle que le reste de l'application ; toutes les variantes sont prédites en un lot (0,01 s par client).
- **Hypothèses affichées** : factures des mois à venir au niveau du dernier mois ; R couvre au moins le minimum exigé (inconnu dans les données) ; reste inchangé. Codes de statut : -1 payé en totalité (documenté) ; 0 payé sans retard, solde reporté ; -2 rien à payer (non documentés ; cohérents avec les données : code 0 → remboursement inférieur à la facture dans 95 % des cas).
- **Réponses honnêtes** : `none_found` (avec le risque le plus bas atteignable), `already_accepted`, `no_actionable`.
- Un mode « maintenant » (liste de variables modifiables déclarée) reste disponible pour d'autres jeux et par l'API.

Exemples : client n°12 (risque 34,8 %) → « dans 3 mois : au moins 11 000 par mois → 15,4 % » ; « dans 6 mois : 1 100 par mois, moins qu'aujourd'hui (1 859), l'essentiel est d'éviter tout retard → 16,6 % ». Client n°850 (94 %) → aucun conseil : au mieux 19,2 % ; sa limite de 10 000 (médiane 140 000) fait alors 60 % de la poussée vers le refus (14,2 % avec une limite médiane), ce qui relève de la banque, pas du client.

### B3 Interface
Onglet Explications : vue « Explication d'un individu » (bouton « Motifs de la décision et conseil ») et vue « Audit du recours ».

### B4 Audit du recours : `recourse_audit.py`, `POST /api/decision/audit`
Le recours est lancé pour chaque client refusé ; on mesure la couverture, l'effort et l'équité de l'accès (part avec conseil par groupe sensible, groupes d'au moins 30, intervalles de Wilson à 95 %, alerte au-delà de 10 points d'écart).

**Résultats (seuil 0,167 ; 3 166 refusés sur 7 500 ; 42 s)**

| Mesure | Valeur |
|---|---|
| reçoivent un conseil | **76,7 %** (2 428) |
| délai le plus court qui suffit : 1 / 3 / 6 mois | 912 / 878 / 638 |
| montant demandé, médiane | × 1,4 le remboursement actuel (saut médian de 7 centiles) |
| sans conseil | 23,3 % (738), dont 58 % sans facture le dernier mois |
| accès par sexe | femmes 75,8 %, hommes 77,8 % |
| accès par situation maritale | 74,5 % à 78,8 % |
| accès par âge | 21-28 : 83,9 % ; 29-34 : 74,0 % ; 35-41 : 71,7 % ; 42-75 : 75,2 % → **alerte** (12,2 points) |

**Lecture.** Les clients sans conseil sont surtout des clients qui utilisent peu leur carte : il n'y a rien à rembourser, et l'application ne conseille jamais de dépenser. Les données donnent en partie raison au modèle (défaut 24,7 % quand la dernière facture est nulle contre 21,9 % ; 35,7 % quand les six factures sont nulles). L'écart selon l'âge passe en grande partie par cette variable proxy (cartes inutilisées : 18 % des refusés de 29 à 41 ans, 10 % des 21-28 ans) : sur les seuls clients ayant une facture, il tombe à 5,8 points. Le reste est à examiner (le modèle utilise l'âge directement).
Avec la première version du recours (modification des montants passés), la couverture était de 58,4 %, et les clients sans conseil étaient les plus risqués (risque médian 64 %).

## Tests
`backend/tests/` : 95 tests réussis, 1 ignoré (fichier de fraude absent). Ceux de la phase B prouvent notamment :
- l'additivité SHAP, et qu'une variable sensible n'est jamais un motif même quand elle domine (jeu fabriqué) ;
- que la fenêtre glisse correctement et que le passé n'est jamais réécrit ;
- que chaque conseil est re-vérifié indépendamment et qu'il est le **plus petit** montant qui suffit (tous les montants plus petits échouent) ;
- que l'audit compte exactement ce que verrait chaque client, et qu'une inégalité d'accès due à une variable fixe liée au sexe (jeu fabriqué) déclenche l'alerte ;
- le tout sur le vrai fichier de Taïwan (test d'intégration, ignoré si le fichier est absent).
`frontend/tests/smoke.test.jsx` parcourt l'interface, cartes de décision et audit compris.

## Limites connues
- Les conseils décrivent le comportement du **modèle**, qui a appris des corrélations ; ils ne garantissent ni le comportement réel du client ni la décision de la banque.
- Les coûts 5:1 sont une hypothèse ; les factures futures et le minimum exigé sont des hypothèses écrites sous chaque conseil.
- Les codes de statut 0 et -2 ne sont pas documentés par l'UCI.
- L'ancien onglet Évaluation affiche encore les notes de test de tous les modèles.
- Le jeu de fraude (phase C) n'a pas encore été testé sur le vrai fichier.
