import { useState } from 'react'
import { useApp } from '../App.jsx'
import { post } from '../api.js'
import { Card, Table, Notice, useAction, Status, Stat, num, pct } from '../ui.jsx'
import Chart from '../Chart.jsx'

// Phase F : le PSI (Population Stability Index) compare les nouveaux clients à ceux de l'apprentissage.
const COLOR = { stable: '#16a34a', 'à surveiller': '#d97706', 'dérive': '#dc2626' }

export default function DriftCard() {
  const { summary, trained, threshold } = useApp()
  const numeric = (summary?.columns || []).filter((c) => c.kind === 'numeric' && c.name !== summary.target).map((c) => c.name)
  const names = trained.map((m) => m.name)                       // modèles entraînés (objets {name, label…})
  const model = names.includes('xgb') ? 'xgb' : names[0] || null
  const [scenario, setScenario] = useState('test')
  const [feature, setFeature] = useState(numeric[0] || '')
  const [factor, setFactor] = useState(1)
  const [offset, setOffset] = useState(1)
  const [rep, setRep] = useState(null)
  const [st, run] = useAction()
  const go = () => run(async () => setRep(await post('/api/drift', {
    model, threshold: Math.min(0.99, Math.max(0.01, threshold)), scenario,
    feature: scenario === 'shifted' ? feature : null, factor: Number(factor), offset: Number(offset),
  })))
  const top = rep ? rep.features.slice(0, 15) : []

  return (
    <Card title="Suivi de dérive : les nouveaux clients ressemblent-ils à ceux de l'apprentissage ?">
      <p className="muted">
        Un modèle apprend le monde d'hier. Si les clients changent, ses probabilités deviennent fausses sans aucun message d'erreur.
        Le PSI (indice de stabilité d'une population) mesure l'écart entre deux distributions d'une même variable :
        moins de 0,10 = stable ; de 0,10 à 0,25 = à surveiller ; plus de 0,25 = dérive (convention du crédit).
        On le calcule pour chaque variable et pour le score du modèle{model ? ` (${model})` : ' (entraînez un modèle pour l\'avoir)'}.
      </p>
      <div className="row wrap">
        <label>Lot à comparer à l'apprentissage
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            <option value="test">le lot de test</option>
            <option value="shifted">le lot de test, volontairement décalé</option>
          </select></label>
        {scenario === 'shifted' && (
          <>
            <label>Variable<select value={feature} onChange={(e) => setFeature(e.target.value)}>
              {numeric.map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
            <label>× facteur<input type="number" step="0.1" value={factor} onChange={(e) => setFactor(e.target.value)} /></label>
            <label>+ décalage<input type="number" step="1" value={offset} onChange={(e) => setOffset(e.target.value)} /></label>
          </>
        )}
        <button className="primary" onClick={go} disabled={st.loading}>Mesurer la dérive</button>
      </div>
      {scenario === 'shifted' && <p className="muted small">Exemple parlant sur Taïwan : PAY_1 × 1 + 1 (chaque client a un mois de retard de plus le mois dernier).</p>}
      <Status state={st} busy="Calcul du PSI de chaque variable…" />
      {rep && (
        <>
          <Notice kind={rep.alert ? 'warn' : 'info'}>{rep.summary}</Notice>
          {rep.score && (
            <div className="stats">
              <Stat label="PSI du score du modèle" value={num(rep.score.psi, 3)} hint={rep.score.level} />
              <Stat label="risque moyen annoncé" value={`${pct(rep.score.mean_reference, 1)} → ${pct(rep.score.mean_current, 1)}`} hint="apprentissage → lot comparé" />
              <Stat label={`taux de refus au seuil ${pct(rep.score.threshold, 1)}`} value={`${pct(rep.score.refusal_rate_reference, 1)} → ${pct(rep.score.refusal_rate_current, 1)}`} />
              <Stat label="variables en dérive / à surveiller" value={`${rep.n_drift} / ${rep.n_watch}`} />
            </div>
          )}
          <Chart height={280} data={[{ type: 'bar', x: top.map((r) => r.feature), y: top.map((r) => r.psi), marker: { color: top.map((r) => COLOR[r.level]) } }]}
            layout={{ yaxis: { title: { text: 'PSI' } }, shapes: [0.1, 0.25].map((y) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y, line: { dash: 'dot', color: '#64748b' } }))}} />
          <Table maxHeight={260} columns={[
            { key: 'feature', label: 'variable' },
            { key: 'psi', label: 'PSI', num: true, render: (r) => num(r.psi, 3) },
            { key: 'level', label: 'niveau', render: (r) => <span style={{ color: COLOR[r.level], fontWeight: 600 }}>{r.level}</span> },
            { key: 'ref_median', label: 'médiane apprentissage', num: true, render: (r) => (r.ref_median == null ? '–' : num(r.ref_median, 2)) },
            { key: 'cur_median', label: 'médiane lot comparé', num: true, render: (r) => (r.cur_median == null ? '–' : num(r.cur_median, 2)) },
          ]} rows={rep.features} />
          <p className="muted small">
            Une variable peut dériver sans que le score bouge, si le modèle s'en sert peu ; c'est le PSI du score qui dit si les décisions changent.
            Une dérive n'est pas une erreur : c'est un signal pour revérifier le modèle (calibration, seuil), voire le réentraîner.
          </p>
        </>
      )}
    </Card>
  )
}
