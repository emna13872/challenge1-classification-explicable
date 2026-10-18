import { useMemo, useState } from 'react'
import { useApp } from '../App.jsx'
import Chart, { COLORS, POS, NEG } from '../Chart.jsx'
import { Card, Table, Notice, Slider, num, pct, Stat } from '../ui.jsx'

// Métriques recalculées côté client à partir du tableau tp/fp/fn/tn (pas d'appel serveur : le curseur est instantané).
function atThr(table, thr) {
  const i = Math.min(table.rows.length - 1, Math.max(0, Math.round(thr * 100)))
  const [, tp, fp, fn, tn] = table.rows[i]
  const precision = tp + fp ? tp / (tp + fp) : 0
  const recall = tp + fn ? tp / (tp + fn) : 0
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0
  return { tp, fp, fn, tn, precision, recall, f1, accuracy: (tp + tn) / (tp + fp + fn + tn) }
}

export default function EvalTab() {
  const { evalData: ev, threshold, setThreshold, goTo } = useApp()
  const names = ev ? Object.keys(ev.models) : []
  const best = useMemo(() => names.slice().sort((a, b) => ev.models[b].auc - ev.models[a].auc)[0], [ev])
  const [sel, setSel] = useState(null)
  const [credit, setCredit] = useState(false)
  if (!ev) return <div className="busy"><span className="spin" /> Évaluation…</div>
  const cur = sel && ev.models[sel] ? sel : best
  const m = ev.models[cur]

  const rows = names.map((n) => ({ name: n, ...ev.models[n], ...atThr(ev.models[n].table, threshold) }))
    .sort((a, b) => b.auc - a.auc)
  const bestAuc = Math.max(...rows.map((r) => r.auc))
  const c = atThr(m.table, threshold)
  const tcurve = m.table.rows.map((r, i) => ({ t: r[0], ...atThr(m.table, r[0]) }))
  const vline = { type: 'line', yref: 'paper', y0: 0, y1: 1, x0: threshold, x1: threshold, line: { color: '#111827', dash: 'dot' } }

  return (
    <div className="stack">
      <Card title="Seuil de décision">
        <div className="row wrap">
          <Slider label="Seuil" value={threshold} min={0} max={1} step={0.01} onChange={setThreshold} fmt={(v) => v.toFixed(2)} />
          <button onClick={() => setThreshold(Math.round(m.best_f1_threshold * 100) / 100)}>Seuil maximisant le F1 ({cur}) : {m.best_f1_threshold.toFixed(2)}</button>
        </div>
        <p className="muted">Un individu est classé « positif » si sa probabilité prédite est ≥ seuil. Baisser le seuil augmente le rappel (on détecte plus de cas) au prix de la précision (plus de fausses alertes).</p>
        <div className="stats">
          <Stat label={`rappel (${m.label})`} value={pct(c.recall)} />
          <Stat label="précision" value={pct(c.precision)} />
          <Stat label="F1" value={num(c.f1)} />
          <Stat label="cas détectés / fausses alertes" value={`${c.tp} / ${c.fp}`} />
        </div>
      </Card>

      <Card title="Tableau comparatif des modèles" right={<label className="inline small"><input type="checkbox" checked={credit} onChange={(e) => setCredit(e.target.checked)} /> Afficher Gini et KS (scoring de crédit)</label>}>
        {ev.imbalanced && <Notice kind="info">Classes déséquilibrées ({pct(ev.positive_rate)} de positifs dans le test) : lisez la PR-AUC, le rappel et le F1 plutôt que l'exactitude.</Notice>}
        <Table maxHeight={320} columns={[
          { key: 'label', label: 'modèle' },
          { key: 'auc', label: 'AUC-ROC', num: true, render: (r) => num(r.auc) },
          { key: 'pr_auc', label: 'PR-AUC', num: true, render: (r) => num(r.pr_auc) },
          ...(credit ? [{ key: 'gini', label: 'Gini', num: true, render: (r) => num(r.gini) }, { key: 'ks', label: 'KS', num: true, render: (r) => num(r.ks) }] : []),
          { key: 'precision', label: 'précision', num: true, render: (r) => num(r.precision) },
          { key: 'recall', label: 'rappel', num: true, render: (r) => num(r.recall) },
          { key: 'f1', label: 'F1', num: true, render: (r) => num(r.f1) },
          { key: 'accuracy', label: 'exactitude', num: true, render: (r) => num(r.accuracy) },
          { key: 'train_seconds', label: 'temps (s)', num: true },
        ]} rows={rows.map((r) => ({ ...r, _cls: r.auc === bestAuc ? 'best' : '' }))} />
        <p className="muted">Précision, rappel et F1 sont mesurés sur la classe d'intérêt au seuil {threshold.toFixed(2)}. Ligne surlignée : meilleure AUC.</p>
      </Card>

      <div className="grid2">
        <Card title="Courbes ROC superposées">
          <Chart height={340} data={[
            ...rows.map((r, i) => ({ type: 'scatter', mode: 'lines', name: `${r.label} (AUC ${r.auc.toFixed(3)})`, x: r.roc.fpr, y: r.roc.tpr, line: { color: COLORS[i % COLORS.length], width: 2 } })),
            { type: 'scatter', mode: 'lines', name: 'hasard', x: [0, 1], y: [0, 1], line: { dash: 'dash', color: '#94a3b8' } }]}
            layout={{ xaxis: { title: 'taux de faux positifs', range: [0, 1] }, yaxis: { title: 'taux de vrais positifs', range: [0, 1.02] } }} />
        </Card>
        <Card title="Courbes précision-rappel">
          <Chart height={340} data={[
            ...rows.map((r, i) => ({ type: 'scatter', mode: 'lines', name: `${r.label} (PR-AUC ${r.pr_auc.toFixed(3)})`, x: r.pr.recall, y: r.pr.precision, line: { color: COLORS[i % COLORS.length], width: 2 } })),
            { type: 'scatter', mode: 'lines', name: 'référence (taux de base)', x: [0, 1], y: [ev.positive_rate, ev.positive_rate], line: { dash: 'dash', color: '#94a3b8' } }]}
            layout={{ xaxis: { title: 'rappel', range: [0, 1] }, yaxis: { title: 'précision', range: [0, 1.02] } }} />
        </Card>
      </div>

      <Card title={`Détail du modèle : ${m.label}`} right={
        <select value={cur} onChange={(e) => setSel(e.target.value)}>{names.map((n) => <option key={n} value={n}>{ev.models[n].label}</option>)}</select>}>
        <div className="grid2">
          <div>
            <h4>Matrice de confusion (seuil {threshold.toFixed(2)})</h4>
            <div className="cm">
              <div /><div className="h">prédit négatif</div><div className="h">prédit positif</div>
              <div className="h">réel négatif</div><div className="cell ok">{c.tn}<small>vrais négatifs</small></div><div className="cell bad">{c.fp}<small>faux positifs</small></div>
              <div className="h">réel positif</div><div className="cell bad">{c.fn}<small>faux négatifs</small></div><div className="cell ok">{c.tp}<small>vrais positifs</small></div>
            </div>
          </div>
          <div>
            <h4>Précision, rappel et F1 selon le seuil</h4>
            <Chart height={280} data={[
              { type: 'scatter', mode: 'lines', name: 'précision', x: tcurve.map((r) => r.t), y: tcurve.map((r) => r.precision), line: { color: COLORS[0] } },
              { type: 'scatter', mode: 'lines', name: 'rappel', x: tcurve.map((r) => r.t), y: tcurve.map((r) => r.recall), line: { color: POS } },
              { type: 'scatter', mode: 'lines', name: 'F1', x: tcurve.map((r) => r.t), y: tcurve.map((r) => r.f1), line: { color: '#16a34a' } }]}
              layout={{ xaxis: { title: 'seuil', range: [0, 1] }, yaxis: { range: [0, 1.02] }, shapes: [vline] }} />
          </div>
        </div>
      </Card>
      <div className="right"><button className="primary" onClick={() => goTo('xai')}>Expliquer les décisions →</button></div>
    </div>
  )
}
