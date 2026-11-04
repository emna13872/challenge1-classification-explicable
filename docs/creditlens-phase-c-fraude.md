# CreditLens : phase C, détection de fraude

Jeu : *Credit Card Fraud Detection* (ULB / Worldline), 284 807 transactions de cartes européennes sur 48 heures (septembre 2013), 492 fraudes (0,17 %). V1 à V28 sont des composantes d'ACP anonymisées ; seuls `Time` et `Amount` sont lisibles. Fichier local attendu : `data/creditcard.csv` ou `data/creditlens/creditcard.csv`.

## Protocole
- **Découpage temporel** (`prep.build`, option `time_column`) : apprentissage = 75 % des transactions les plus anciennes (213 605, 398 fraudes) ; test = les 9,3 dernières heures (71 202, 94 fraudes). `Time` n'est pas une variable du modèle. Aucun des 1 081 doublons exacts n'est coupé entre apprentissage et test.
- **Validation glissante** (`fraud.forward_oof`, `TimeSeriesSplit` à 4 coupures) : chaque bloc de temps de l'apprentissage est noté par un modèle appris sur les blocs antérieurs (préparation réapprise à chaque fois) ; 170 884 transactions notées, 278 fraudes.
- **Coûts** : une fraude manquée coûte son montant ; une alerte coûte une vérification `review_cost` (hypothèse : 5). Coût d'une stratégie = montants laissés passer + `review_cost` × alertes.
- **Stratégies comparées** : ne rien vérifier ; seuil 50 % ; meilleur seuil unique (optimisé sur la validation) ; règle du montant `p × montant > review_cost` ; budgets de 0,5, 1, 2, 5 alertes pour 1 000 transactions.
- **Choix** sur l'apprentissage seul (variante de modèle et stratégie de coût minimal) ; puis entraînement sur tout l'apprentissage et **ouverture unique du test**, comptée (`Session.test_openings`), avec intervalle bootstrap à 95 % du coût pour 1 000 transactions.

## Résultats

**Validation glissante (apprentissage)**

| Variante | PR-AUC | AUC | proba. moyenne / taux réel | évitées : seuil 50 % | évitées : meilleur seuil | évitées : règle du montant |
|---|---|---|---|---|---|---|
| Régression logistique | 0,622 | 0,946 | × 0,9 | 31,3 % | 55,4 % | 63,5 % |
| Régression logistique rééquilibrée | 0,747 | 0,955 | × 37,9 | 24,6 % | 65,5 % | −184,4 % |
| XGBoost | 0,788 | 0,965 | × 0,9 | 65,8 % | 69,3 % | **71,0 %** |
| XGBoost rééquilibré | 0,783 | 0,947 | × 2,2 | 68,7 % | 69,2 % | 64,4 % |

Retenu : XGBoost + règle du montant. Le rééquilibrage de la régression logistique améliore la PR-AUC mais gonfle les probabilités : la règle du montant déclenche alors 127 alertes pour 1 000 transactions et coûte 2,8 fois plus que ne rien vérifier.

**Test (ouvert une fois)**

| Stratégie | alertes / 1 000 | rappel | précision | pertes évitées |
|---|---|---|---|---|
| Ne rien vérifier | 0 | 0 % | – | 0 % |
| Seuil 50 % | 0,98 | 71 % | 96 % | 71,9 % |
| Seuil choisi sur l'apprentissage (0,053) | 1,53 | 79 % | 68 % | 72,8 % |
| **Règle du montant (retenue)** | 1,67 | 36 % | 29 % | **77,7 %** |
| Budget 1 / 1 000 | 1,00 | 71 % | 94 % | 71,9 % |
| Budget 5 / 1 000 | 5,00 | 84 % | 22 % | 70,7 % |

PR-AUC 0,800. Coût pour 1 000 transactions : 33,9 (IC 95 % : 10,6 à 66,8) ; la validation promettait 65,2, dans l'intervalle. La règle du montant n'attrape que 36 % des fraudes en nombre mais 83 % du montant fraudé : 45 % des fraudes font 5 ou moins et ne pèsent que 0,4 % de l'argent. Le taux de fraude passe de 0,19 % (apprentissage) à 0,13 % (test).

## Ce qui ne se transpose pas du crédit
- **Pas de recours** : `recourse` répond `not_applicable` pour ce jeu (un conseil apprendrait à contourner le modèle) ; l'audit du recours aussi.
- **Motifs** : SHAP reste calculé, mais une note rappelle que V1 à V28 n'ont pas de sens métier.

## Tests
`backend/tests/test_fraud.py` (11 tests sur données synthétiques) : test entièrement postérieur à l'apprentissage ; premier bloc jamais noté ; bilans exacts sur exemples calculés à la main ; meilleur seuil ; rééquilibrage → probabilités gonflées et explosion des alertes ; choix inchangé quand on inverse les étiquettes du test ; test ouvert une fois puis non relu ; routes. `test_real_datasets.py::test_fraud_end_to_end` rejoue le parcours sur le vrai fichier.

## Limites
Coût de vérification supposé ; les micro-transactions de test de carte ne sont pas valorisées ; 94 fraudes seulement dans le test (intervalle large) ; 48 heures de données ; montants sans devise précisée.
