import { useState } from 'react'
import { post } from '../api.js'
import { Card, Table, Notice, useAction, Status, Stat, num, pct } from '../ui.jsx'
import Chart, { COLORS } from '../Chart.jsx'

// Phase C : quand les fraudes sont rares (0,17 %) et qu'une fraude manquée coûte SON montant.
const money = (v) => (v == null ? '–' : Math.round(v).toLocaleString('fr-FR'))
const STRATS = [['half', 'seuil 50 %'], ['best_threshold', 'meilleur seuil unique'], ['amount_rule', 'règle du montant']]

function outcomeRows(t, labels) {
  return labels.map(([k, label]) => ({ k: label, ...t[k] }))
}

const OUTCOME_COLUMNS = (n) => [
  { key: 'k', label: n },
  { key: 'alerts_per_1000', label: 'alertes pour 1 000', num: true, render: (r) => num(r.alerts_per_1000, 2) },
  { key: 'recall', label: 'fraudes attrapées (rappel)', num: true, render: (r) => pct(r.recall, 0) },
  { key: 'precision', label: 'alertes justifiées (précision)', num: true, render: (r) => pct(r.precision, 0) },
  { key: 'amount_caught', label: 'montant sauvé', num: true, render: (r) => money(r.amount_caught) },
  { key: 'review_cost', label: 'coût des vérifications', num: true, render: (r) => money(r.review_cost) },
  { key: 'saved_share', label: 'pertes évitées', num: true, render: (r) => pct(r.saved_share, 1) },
]

export default function FraudCard() {
  const [cost, setCost] = useState(5)
  const [rep, setRep] = useState(null)
  const [fin, setFin] = useState(null)
  const [st, run] = useAction()
  const [finSt, runFin] = useAction()
  const report = () => run(async () => { setRep(await post('/api/fraud/report', { review_cost: Number(cost) })); setFin(null) })
  const final = (force = false) => runFin(async () => setFin(await post('/api/fraud/final', { review_cost: Number(cost), force })))
  const chosen = rep && rep.variants.find((v) => v.key === rep.choice.key)

  return (
    <Card title="Fraude : quelles transactions vérifier ?">
      <p className="muted">
        Les fraudes sont très rares : ne rien signaler donne déjà une exactitude de 99,8 %, sans arrêter une seule fraude.
        On juge donc autrement. Une fraude manquée coûte <b>son montant</b> ; une alerte coûte une <b>vérification</b>{' '}
        (hypothèse à remplacer par le vrai coût d'un analyste). La « règle du montant » alerte si probabilité × montant &gt; coût
        de vérification. Tout est choisi sur l'apprentissage, en validant <b>dans le temps</b> (on apprend sur le passé, on note la suite).
      </p>
      <div className="row wrap">
        <label>Coût d'une vérification<input type="number" min={0.1} step={0.5} value={cost} onChange={(e) => setCost(e.target.value)} /></label>
        <button className="primary" onClick={report} disabled={st.loading || !(cost > 0)}>Comparer sur l'apprentissage</button>
      </div>
      <Status state={st} busy="Validation dans le temps de 4 variantes de modèles (environ 30 s sur le vrai fichier)…" />
      {rep && (
        <>
          {!rep.settings.temporal && <Notice kind="warn">Pas de colonne de temps dans la préparation : validation au hasard, moins réaliste pour la fraude.</Notice>}
          <Table maxHeight={260} columns={[
            { key: 'label', label: 'modèle' },
            { key: 'pr_auc', label: 'PR-AUC', num: true, render: (r) => num(r.pr_auc, 3) },
            { key: 'cal', label: 'proba. moyenne / taux réel', num: true, render: (r) => `× ${num(r.calibration.ratio, 1)}` },
            ...STRATS.map(([k, label]) => ({ key: k, label: `évitées : ${label}`, num: true, render: (r) => pct(r.strategies[k].saved_share, 1) })),
          ]} rows={rep.variants} />
          <p className="muted small">
            « Évitées » : part des pertes dues à la fraude évitée par chaque stratégie, vérifications payées. Pour comparaison, ne rien
            signaler aurait une exactitude de {pct(rep.variants[0].accuracy_accept_all, 2)}, pour 0 % de pertes évitées.
          </p>
          <Notice kind="info">
            Retenu sur l'apprentissage : <b>{rep.choice.label}</b> + <b>{rep.choice.strategy_label.toLowerCase()}</b>, qui évite{' '}
            {pct(rep.choice.expected.saved_share, 1)} des pertes ({rep.settings.n_scored.toLocaleString('fr-FR')} transactions notées,
            dont {rep.settings.n_frauds_scored} fraudes). Une probabilité moyenne loin de × 1 veut dire des probabilités fausses :
            la règle du montant devient alors absurde (trop d'alertes).
          </Notice>
          {chosen && (
            <>
              <h4>Et si l'équipe ne peut vérifier qu'un nombre fixe d'alertes ? ({chosen.label})</h4>
              <div className="grid2">
                <Table maxHeight={220} columns={[
                  { key: 'per_1000', label: 'alertes pour 1 000 transactions', num: true },
                  { key: 'recall', label: 'fraudes attrapées', num: true, render: (r) => pct(r.recall, 0) },
                  { key: 'precision', label: 'alertes justifiées', num: true, render: (r) => pct(r.precision, 0) },
                  { key: 'saved_share', label: 'pertes évitées', num: true, render: (r) => pct(r.saved_share, 1) }]} rows={chosen.strategies.budgets} />
                <Chart height={220} data={[
                  { x: chosen.strategies.budgets.map((b) => b.per_1000), y: chosen.strategies.budgets.map((b) => b.recall * 100), mode: 'lines+markers', name: 'fraudes attrapées (%)', line: { color: COLORS[0] } },
                  { x: chosen.strategies.budgets.map((b) => b.per_1000), y: chosen.strategies.budgets.map((b) => b.precision * 100), mode: 'lines+markers', name: 'alertes justifiées (%)', line: { color: COLORS[2] } },
                ]} layout={{ legend: { orientation: 'h', y: 1.18 }, margin: { l: 45, r: 10, t: 40, b: 45 },
                  xaxis: { title: { text: 'alertes pour 1 000 transactions' } }, yaxis: { title: { text: '%' }, range: [0, 105] } }} />
              </div>
            </>
          )}
          <div className="row">
            <button className="primary" onClick={() => final(false)} disabled={finSt.loading}>{fin ? 'Revoir le résultat du test (fraude)' : 'Ouvrir le lot de test pour la fraude (une seule fois)'}</button>
            {fin && <button onClick={() => final(true)} disabled={finSt.loading}>Recalculer et rouvrir le test (compté)</button>}
          </div>
          <Status state={finSt} busy="Entraînement final puis lecture du lot de test…" />
        </>
      )}
      {fin && (
        <>
          <Notice kind={fin.openings > 1 ? 'warn' : 'info'}>
            Promesse de la validation : <b>{pct(fin.expected.saved_share, 1)}</b> des pertes évitées ({num(fin.expected.cost_per_1000, 1)} de coût pour 1 000 transactions).
            Sur le lot de test (les transactions les plus récentes) : <b>{pct(fin.test.chosen.saved_share, 1)}</b> ({num(fin.test.chosen.cost_per_1000, 1)} pour 1 000 ;
            intervalle à 95 % : {num(fin.test.cost_per_1000_ci95[0], 1)} à {num(fin.test.cost_per_1000_ci95[1], 1)}).
            {fin.test.expected_inside_ci ? ' La promesse est dans l\'intervalle.' : ' La promesse sort de l\'intervalle : à examiner.'}
            {' '}Lot de test ouvert {fin.openings} fois{fin.already_opened ? ' (résultat déjà obtenu, le test n\'a pas été relu)' : ''}.{fin.warning ? ' ' + fin.warning : ''}
          </Notice>
          <div className="stats">
            <Stat label="PR-AUC sur le test" value={num(fin.test.pr_auc, 3)} />
            <Stat label="taux de fraude : apprentissage → test" value={`${pct(fin.train_base_rate, 2)} → ${pct(fin.test.base_rate, 2)}`} hint="le monde change d'un jour à l'autre" />
            <Stat label="fraudes dans le test" value={fin.test.chosen.frauds_total} hint="peu nombreuses : intervalle large" />
          </div>
          <Table maxHeight={420} columns={OUTCOME_COLUMNS(`sur les ${fin.test.n.toLocaleString('fr-FR')} transactions du test`)} rows={[
            ...outcomeRows(fin.test, [['accept_all', 'Ne rien vérifier'], ['half', 'Seuil fixe de 50 %'], ['train_threshold', `Seuil choisi sur l'apprentissage (${num(fin.test.train_threshold.threshold, 3)})`], ['amount_rule', 'Règle du montant']]),
            ...fin.test.budgets.map((b) => ({ k: `Budget : ${b.per_1000} alerte(s) pour 1 000`, ...b })),
          ]} />
          <p className="muted small">
            Choix retenu : {fin.choice.label} + {fin.choice.strategy_label.toLowerCase()}. Les petites fraudes (montant inférieur au coût d'une vérification)
            ne sont jamais vérifiées par la règle du montant : c'est voulu, mais une transaction de quelques centimes est parfois un test de carte avant une grosse fraude,
            ce que ce calcul de coût ne voit pas.
          </p>
        </>
      )}
    </Card>
  )
}
