import { useEffect, useState } from 'react'
import { useApp } from '../App.jsx'
import { get, post, upload } from '../api.js'
import Chart, { COLORS, POS, NEG } from '../Chart.jsx'
import { Card, Stat, Table, Notice, pct, num, useAction, Status } from '../ui.jsx'

export default function DataTab() {
  const { summary, setSummary, goTo } = useApp()
  const [datasets, setDatasets] = useState([])
  const [ds, setDs] = useState('telco')
  const [csvTarget, setCsvTarget] = useState('')
  const [st, run] = useAction()

  useEffect(() => { get('/api/datasets').then(setDatasets).catch(() => {}) }, [])

  const load = () => run(async () => setSummary(await post('/api/data/load', { dataset: ds })))
  const onFile = (e) => {
    const f = e.target.files[0]
    if (!f) return
    const fd = new FormData()
    fd.append('file', f)
    if (csvTarget.trim()) fd.append('target', csvTarget.trim())
    run(async () => setSummary(await upload('/api/data/upload', fd)))
    e.target.value = ''
  }
  const setTarget = (target, positive) => run(async () => setSummary(await post('/api/data/target', { target, positive })))

  return (
    <div className="stack">
      <Card title="Source des données">
        <div className="row wrap">
          <label>Jeu de données public (téléchargement automatique)
            <select value={ds} onChange={(e) => setDs(e.target.value)}>
              {datasets.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
            </select>
          </label>
          <button className="primary" onClick={load} disabled={st.loading}>Télécharger et charger</button>
          <span className="sep">ou</span>
          <label>Variable cible du CSV (optionnel, sinon dernière colonne)
            <input value={csvTarget} onChange={(e) => setCsvTarget(e.target.value)} placeholder="ex. Churn" />
          </label>
          <label className="file">Importer un CSV<input type="file" accept=".csv,text/csv" onChange={onFile} /></label>
        </div>
        <Status state={st} busy="Téléchargement / lecture des données…" />
        {datasets.find((d) => d.key === ds) && <p className="muted">Source : {datasets.find((d) => d.key === ds).source}</p>}
      </Card>

      {summary && <>
        <Summary summary={summary} setTarget={setTarget} busy={st.loading} />
        <Explore summary={summary} />
        <div className="right"><button className="primary" onClick={() => goTo('prep')}>Passer à la préparation →</button></div>
      </>}
    </div>
  )
}

function Summary({ summary: s, setTarget, busy }) {
  const missing = s.columns.filter((c) => c.missing > 0)
  return (
    <>
      <Card title="Aperçu et variable cible">
        <div className="stats">
          <Stat label="observations" value={s.n_rows.toLocaleString('fr-FR')} />
          <Stat label="colonnes" value={s.n_cols} />
          <Stat label={`classe « ${s.positive} »`} value={pct(s.positive_rate)} hint={s.imbalanced ? 'classes déséquilibrées' : 'classes équilibrées'} />
          <Stat label="colonnes avec valeurs manquantes" value={missing.length} />
        </div>
        <div className="row wrap">
          <label>Variable cible
            <select value={s.target} disabled={busy} onChange={(e) => setTarget(e.target.value)}>
              {s.columns.map((c) => <option key={c.name}>{c.name}</option>)}
            </select>
          </label>
          <label>Classe d'intérêt (positive)
            <select value={s.positive} disabled={busy} onChange={(e) => setTarget(s.target, e.target.value)}>
              {s.classes.map((c) => <option key={c.label} value={c.label}>{c.label} ({c.count})</option>)}
            </select>
          </label>
        </div>
        {s.multiclass && <Notice kind="warn">La cible a plus de deux classes : la modalité choisie est opposée à toutes les autres (binarisation).</Notice>}
        {s.imbalanced && <Notice kind="info">Classes déséquilibrées : privilégiez le rappel, le F1 et la PR-AUC à l'exactitude.</Notice>}
        <Chart height={180} data={[{ type: 'bar', orientation: 'h', y: s.classes.map((c) => c.label), x: s.classes.map((c) => c.pct),
          marker: { color: s.classes.map((c) => (c.label === s.positive ? POS : '#94a3b8')) },
          text: s.classes.map((c) => `${c.pct.toFixed(1)} %`), textposition: 'auto' }]}
          layout={{ xaxis: { title: 'proportion (%)' }, margin: { l: 90, r: 20, t: 10, b: 40 } }} />
      </Card>
      <Card title="Aperçu des 20 premières lignes">
        <Table maxHeight={300} columns={s.columns.map((c) => ({ key: c.name, label: c.name }))} rows={s.preview} />
      </Card>
      <div className="grid2">
        <Card title="Statistiques descriptives (variables numériques)">
          <Table maxHeight={300} columns={[
            { key: 'column', label: 'variable' }, { key: 'mean', label: 'moy.', num: true }, { key: 'std', label: 'éc.-type', num: true },
            { key: 'min', label: 'min', num: true }, { key: '50%', label: 'médiane', num: true }, { key: 'max', label: 'max', num: true }]}
            rows={s.describe} />
        </Card>
        <Card title="Valeurs manquantes">
          {missing.length === 0 ? <p className="muted">Aucune valeur manquante.</p> :
            <Table maxHeight={300} columns={[{ key: 'name', label: 'variable' }, { key: 'missing', label: 'manquantes', num: true },
              { key: 'p', label: '%', num: true, render: (r) => r.missing_pct.toFixed(2) }]} rows={missing} />}
        </Card>
      </div>
    </>
  )
}

function Explore({ summary: s }) {
  const [col, setCol] = useState(s.columns.find((c) => c.name !== s.target)?.name)
  const [view, setView] = useState(null)
  const [corr, setCorr] = useState(null)
  const [st, run] = useAction()
  const [st2, run2] = useAction()

  useEffect(() => { setCorr(null) }, [s.name, s.target, s.positive])
  useEffect(() => {
    if (!col) return
    run(async () => setView(await get('/api/data/column/' + encodeURIComponent(col))))
  }, [col, s.target, s.positive])

  return (
    <Card title="Exploration interactive">
      <div className="row wrap">
        <label>Variable à explorer
          <select value={col} onChange={(e) => setCol(e.target.value)}>{s.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select>
        </label>
        <button onClick={() => run2(async () => setCorr(await get('/api/data/correlation')))} disabled={st2.loading}>Matrice de corrélation</button>
      </div>
      <Status state={st} /><Status state={st2} />
      {view && (
        <div className="grid2">
          <Chart height={300} data={[{ type: 'bar', x: view.kind === 'numeric' ? view.hist.map((h) => (h.x0 + h.x1) / 2) : view.hist.map((h) => h.label),
            y: view.hist.map((h) => h.count), marker: { color: COLORS[0] } }]}
            layout={{ title: { text: 'Distribution de ' + view.column, font: { size: 13 } }, yaxis: { title: 'effectif' } }} />
          <Chart height={300} data={[{ type: 'bar', x: view.rate.map((r) => r.label), y: view.rate.map((r) => r.rate * 100),
            marker: { color: COLORS[1] }, customdata: view.rate.map((r) => r.count),
            hovertemplate: '%{x}<br>taux : %{y:.1f} %<br>effectif : %{customdata}<extra></extra>' }]}
            layout={{ title: { text: `Taux de « ${s.positive} » par ${view.kind === 'numeric' ? 'tranche' : 'modalité'}`, font: { size: 13 } },
              yaxis: { title: '% classe positive' }, xaxis: { tickangle: -30 },
              shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: view.base_rate * 100, y1: view.base_rate * 100, line: { dash: 'dash', color: POS } }] }} />
        </div>
      )}
      {corr && (
        <div className="grid2">
          <Chart height={420} data={[{ type: 'heatmap', z: corr.matrix, x: corr.labels, y: corr.labels, zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true }]}
            layout={{ title: { text: 'Corrélations (variables numériques et cible)', font: { size: 13 } }, margin: { l: 110, b: 110, t: 40, r: 10 } }} />
          <Chart height={420} data={[{ type: 'bar', orientation: 'h',
            y: [...corr.target_corr].sort((a, b) => Math.abs(a.corr) - Math.abs(b.corr)).map((c) => c.feature),
            x: [...corr.target_corr].sort((a, b) => Math.abs(a.corr) - Math.abs(b.corr)).map((c) => c.corr),
            marker: { color: [...corr.target_corr].sort((a, b) => Math.abs(a.corr) - Math.abs(b.corr)).map((c) => (c.corr >= 0 ? POS : NEG)) } }]}
            layout={{ title: { text: 'Corrélation avec la cible', font: { size: 13 } }, margin: { l: 130, t: 40, r: 10 } }} />
        </div>
      )}
    </Card>
  )
}
