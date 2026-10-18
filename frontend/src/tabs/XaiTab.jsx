import { useEffect, useMemo, useState } from 'react'
import { useApp } from '../App.jsx'
import { get, post } from '../api.js'
import Chart, { COLORS, POS, NEG } from '../Chart.jsx'
import { Card, Table, Notice, Slider, useAction, Status, pct, num, Stat } from '../ui.jsx'

export default function XaiTab() {
  const { trained, evalData } = useApp()
  const [model, setModel] = useState(null)
  const [view, setView] = useState('global')
  const names = trained.map((m) => m.name)
  const cur = model && names.includes(model) ? model : (evalData ? Object.keys(evalData.models).sort((a, b) => evalData.models[b].auc - evalData.models[a].auc)[0] : names[0])
  const label = (n) => trained.find((m) => m.name === n)?.label || n

  return (
    <div className="stack">
      <Card>
        <div className="row wrap">
          <label>Modèle à expliquer
            <select value={cur} onChange={(e) => setModel(e.target.value)}>{names.map((n) => <option key={n} value={n}>{label(n)}</option>)}</select>
          </label>
          <div className="seg">
            {[['global', 'Explications globales'], ['local', 'Explication d\'un individu'], ['fair', 'Variables sensibles et équité']].map(([k, t]) => (
              <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{t}</button>))}
          </div>
        </div>
      </Card>
      {view === 'global' && <Global model={cur} label={label} names={names} />}
      {view === 'local' && <Local model={cur} />}
      {view === 'fair' && <Fairness model={cur} />}
    </div>
  )
}

// ---------------------------------------------------------------------------------------- global
function Global({ model, label, names }) {
  const [g, setG] = useState(null)
  const [perm, setPerm] = useState(null)
  const [dep, setDep] = useState(0)
  const [st, run] = useAction()
  const [st2, run2] = useAction()

  useEffect(() => {
    setG(null)
    run(async () => setG(await get('/api/explain/global/' + model)))
  }, [model])
  useEffect(() => {
    setPerm(null)
    run2(async () => setPerm(await get('/api/explain/permutation?models=' + names.join(','))))
  }, [names.join(',')])

  const bee = useMemo(() => {
    if (!g) return null
    const f = g.shap.beeswarm.slice(0, 12)
    const n = f.length
    return {
      traces: f.map((b, i) => ({
        type: 'scatter', mode: 'markers', showlegend: false, name: b.feature,
        x: b.shap, y: b.shap.map((_, k) => n - 1 - i + (((k * 9301 + 49297 + i * 131) % 233280) / 233280 - 0.5) * 0.55),
        marker: { size: 6, opacity: 0.8, color: b.color, cmin: 0, cmax: 1, colorscale: [[0, NEG], [1, POS]],
          showscale: i === 0, colorbar: { title: { text: 'valeur' }, tickvals: [0, 1], ticktext: ['faible', 'élevée'], len: 0.6 } },
        text: b.value.map((v) => `${b.feature} = ${Number(v).toFixed(2)}`),
        hovertemplate: '%{text}<br>SHAP : %{x:.3f}<extra></extra>',
      })),
      ticks: f.map((b, i) => n - 1 - i), names: f.map((b) => b.feature),
    }
  }, [g])

  return (
    <>
      <Status state={st} busy="Calcul des valeurs SHAP…" />
      {g && (
        <>
          <div className="grid2">
            <Card title="Importance propre au modèle">
              {!g.native.available ? <Notice kind="info">{g.native.label}</Notice> : (
                <Chart height={420} data={[{ type: 'bar', orientation: 'h',
                  y: g.native.items.map((i) => i.feature).reverse(), x: g.native.items.map((i) => i.value).reverse(),
                  marker: { color: g.native.items.map((i) => (g.native.signed ? (i.value >= 0 ? POS : NEG) : COLORS[0])).reverse() } }]}
                  layout={{ xaxis: { title: g.native.label }, margin: { l: 170, t: 10, r: 10, b: 55 } }} />)}
              {g.native.available && g.native.signed && <p className="muted">Rouge : augmente la probabilité de la classe positive ; bleu : la diminue.</p>}
            </Card>
            <Card title="Importance par permutation (comparée entre modèles)">
              <Status state={st2} busy="Permutation des variables…" />
              {perm && <PermChart perm={perm} label={label} />}
            </Card>
          </div>

          <Card title="SHAP summary (beeswarm)">
            <p className="muted">Chaque point est un individu du jeu de test ({g.shap.n_samples} échantillonnés). Position horizontale : impact sur la sortie du modèle (échelle : {g.shap.space}) ; couleur : valeur de la variable. Variables triées par impact moyen absolu.</p>
            <Chart height={Math.max(380, 36 * bee.names.length + 90)} data={bee.traces}
              layout={{ xaxis: { title: `valeur SHAP (${g.shap.space})`, zeroline: true },
                yaxis: { tickvals: bee.ticks, ticktext: bee.names, zeroline: false, range: [-0.7, bee.names.length - 0.3] }, margin: { l: 190, t: 10, r: 10, b: 55 } }} />
          </Card>

          <Card title="Graphiques de dépendance SHAP" right={
            <select value={dep} onChange={(e) => setDep(parseInt(e.target.value, 10))}>{g.shap.dependence.map((d, i) => <option key={d.feature} value={i}>{d.feature}</option>)}</select>}>
            {g.shap.dependence[dep] && (
              <Chart height={340} data={[{ type: 'scatter', mode: 'markers', x: g.shap.dependence[dep].x, y: g.shap.dependence[dep].shap,
                marker: { color: COLORS[0], opacity: 0.65, size: 7 } }]}
                layout={{ xaxis: { title: g.shap.dependence[dep].feature }, yaxis: { title: `valeur SHAP (${g.shap.space})` },
                  shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: '#94a3b8', dash: 'dash' } }] }} />)}
            <p className="muted">Lecture : au-dessus de 0, la valeur de la variable pousse la prédiction vers la classe positive.</p>
          </Card>
        </>
      )}
    </>
  )
}

function PermChart({ perm, label }) {
  const ms = Object.keys(perm)
  const avg = {}
  ms.forEach((m) => perm[m].items.forEach((i) => { avg[i.feature] = (avg[i.feature] || 0) + i.importance / ms.length }))
  const top = Object.entries(avg).sort((a, b) => b[1] - a[1]).slice(0, 12).map((e) => e[0]).reverse()
  return (
    <Chart height={420} data={ms.map((m, k) => ({
      type: 'bar', orientation: 'h', name: label(m), y: top,
      x: top.map((f) => perm[m].items.find((i) => i.feature === f)?.importance ?? 0), marker: { color: COLORS[k % COLORS.length] },
    }))} layout={{ barmode: 'group', xaxis: { title: "baisse d'AUC-ROC quand la variable est permutée" }, margin: { l: 150, t: 10, r: 10, b: 80 }, legend: { orientation: 'h', y: -0.25 } }} />
  )
}

// ---------------------------------------------------------------------------------------- local
function Local({ model }) {
  const { threshold, setThreshold } = useApp()
  const [mode, setMode] = useState('test')
  const [samples, setSamples] = useState([])
  const [idx, setIdx] = useState(0)
  const [form, setForm] = useState(null)
  const [rec, setRec] = useState({})
  const [res, setRes] = useState(null)
  const [st, run] = useAction()

  useEffect(() => {
    get(`/api/explain/test-samples/${model}?limit=40`).then((s) => { setSamples(s); if (s[0]) setIdx(s[0].index) }).catch(() => {})
    get('/api/explain/form').then((f) => { setForm(f); setRec(Object.fromEntries(f.fields.map((x) => [x.name, x.default]))) }).catch(() => {})
    setRes(null)
  }, [model])

  const explain = () => run(async () => {
    const body = { model, threshold, ...(mode === 'test' ? { test_index: Number(idx) } : { record: rec }) }
    setRes(await post('/api/explain/local', body))
  })

  return (
    <>
      <Card title="Individu à expliquer">
        <div className="seg">
          <button className={mode === 'test' ? 'on' : ''} onClick={() => setMode('test')}>Individu du jeu de test</button>
          <button className={mode === 'form' ? 'on' : ''} onClick={() => setMode('form')}>Individu fictif (formulaire)</button>
        </div>
        {mode === 'test' && (
          <div className="row wrap">
            <label>Parmi les 40 individus les plus à risque
              <select value={idx} onChange={(e) => setIdx(e.target.value)}>
                {samples.map((s) => <option key={s.index} value={s.index}>n°{s.index} · p = {s.proba.toFixed(2)} · réel : {s.true ? 'positif' : 'négatif'}</option>)}
              </select></label>
            <label>ou n° dans le jeu de test<input type="number" min={0} max={form ? form.n_test - 1 : undefined} value={idx} onChange={(e) => setIdx(e.target.value)} /></label>
            <button onClick={() => form && setIdx(Math.floor(Math.random() * form.n_test))}>Au hasard</button>
          </div>
        )}
        {mode === 'form' && form && (
          <div className="formgrid">
            {form.fields.map((f) => (
              <label key={f.name}>{f.name}
                {f.kind === 'cat'
                  ? <select value={rec[f.name]} onChange={(e) => setRec({ ...rec, [f.name]: e.target.value })}>{f.choices.map((c) => <option key={c}>{c}</option>)}</select>
                  : <input type="number" min={f.min} max={f.max} step={f.integer ? 1 : 'any'} value={rec[f.name]}
                      onChange={(e) => setRec({ ...rec, [f.name]: e.target.value === '' ? '' : parseFloat(e.target.value) })} />}
              </label>
            ))}
          </div>
        )}
        <div className="row wrap">
          <Slider label="Seuil de décision" value={threshold} min={0} max={1} step={0.01} onChange={setThreshold} fmt={(v) => v.toFixed(2)} />
          <button className="primary" onClick={explain} disabled={st.loading}>Expliquer cette prédiction</button>
        </div>
        <Status state={st} busy="Calcul de SHAP et LIME…" />
      </Card>
      {res && <LocalResult res={res} />}
    </>
  )
}

function waterfall(w) {
  const rows = w.items.map((i) => ({ label: i.label, v: i.shap }))
  if (w.n_others) rows.push({ label: `${w.n_others} autres variables`, v: w.others })
  let cum = w.base
  const starts = rows.map((r) => { const s = cum; cum += r.v; return s })
  return { rows, starts }
}

function LocalResult({ res }) {
  const w = res.waterfall
  const { rows, starts } = useMemo(() => waterfall(w), [res])
  const lime = res.lime.items
  const ag = res.agreement
  // Marge de part et d'autre pour que les valeurs affichées ne chevauchent pas les noms de variables.
  const pad = (lo, hi) => { const s = (hi - lo) || 1; return [lo - 0.3 * s, hi + 0.3 * s] }
  const wfRange = pad(
    Math.min(w.base, w.output, ...rows.map((r, i) => Math.min(starts[i], starts[i] + r.v))),
    Math.max(w.base, w.output, ...rows.map((r, i) => Math.max(starts[i], starts[i] + r.v))))
  const limeRange = pad(Math.min(0, ...lime.map((l) => l.weight)), Math.max(0, ...lime.map((l) => l.weight)))
  return (
    <>
      <Card title="Prédiction">
        <div className="stats">
          <Stat label="probabilité prédite" value={pct(res.proba)} />
          <Stat label={`décision au seuil ${res.threshold.toFixed(2)}`} value={res.decision ? 'positif (à risque)' : 'négatif'} />
          {res.true_label != null && <Stat label="réalité" value={res.true_label ? 'positif' : 'négatif'} hint={res.true_label === (res.decision ? 1 : 0) ? 'prédiction correcte' : 'prédiction erronée'} />}
        </div>
        <div className={'sentence ' + (res.decision ? 'hi' : 'lo')}>{res.sentence}</div>
      </Card>

      <div className="grid2">
        <Card title="SHAP waterfall">
          <Chart height={Math.max(320, 34 * rows.length + 100)} data={[{
            type: 'bar', orientation: 'h', y: rows.map((r) => r.label), x: rows.map((r) => r.v), base: starts,
            marker: { color: rows.map((r) => (r.v >= 0 ? POS : NEG)) },
            text: rows.map((r) => (r.v >= 0 ? '+' : '') + r.v.toFixed(3)), textposition: 'outside', cliponaxis: false,
            hovertemplate: '%{y}<br>contribution : %{text}<extra></extra>' }]}
            layout={{ yaxis: { autorange: 'reversed', automargin: true }, xaxis: { title: `échelle : ${w.space}`, range: wfRange }, margin: { l: 10, r: 40, t: 20, b: 55 },
              shapes: [
                { type: 'line', yref: 'paper', y0: 0, y1: 1, x0: w.base, x1: w.base, line: { color: '#94a3b8', dash: 'dash' } },
                { type: 'line', yref: 'paper', y0: 0, y1: 1, x0: w.output, x1: w.output, line: { color: '#111827', dash: 'dot' } }],
              annotations: [
                { xref: 'x', yref: 'paper', x: w.base, y: 1.04, text: `base ${w.base.toFixed(2)}`, showarrow: false, font: { size: 11, color: '#64748b' } },
                { xref: 'x', yref: 'paper', x: w.output, y: 1.04, text: `sortie ${w.output.toFixed(2)}`, showarrow: false, font: { size: 11 } }] }} />
          <p className="muted">On part de la valeur moyenne du modèle (base) ; chaque barre est la contribution d'une variable, en rouge si elle augmente le risque, en bleu si elle le diminue.</p>
        </Card>
        <Card title="Explication LIME (même individu)">
          <Chart height={Math.max(320, 34 * lime.length + 100)} data={[{
            type: 'bar', orientation: 'h', y: lime.map((l) => l.label), x: lime.map((l) => l.weight),
            marker: { color: lime.map((l) => (l.weight >= 0 ? POS : NEG)) },
            text: lime.map((l) => (l.weight >= 0 ? '+' : '') + l.weight.toFixed(3)), textposition: 'outside', cliponaxis: false }]}
            layout={{ yaxis: { autorange: 'reversed', automargin: true }, xaxis: { title: 'poids local (probabilité)', range: limeRange }, margin: { l: 10, r: 40, t: 20, b: 55 } }} />
          <p className="muted">Modèle linéaire local ajusté autour de l'individu : le poids indique l'effet de la condition sur la probabilité de la classe positive.</p>
        </Card>
      </div>

      <Card title="Comparaison SHAP / LIME">
        <div className="stats">
          <Stat label="variables communes dans le top 5" value={`${ag.top5_overlap} / 5`} />
          <Stat label="concordance de signe" value={ag.sign_agreement == null ? '–' : pct(ag.sign_agreement, 0)} hint={`${ag.n_common} variables comparées`} />
          <Stat label="corrélation de rang (Spearman)" value={ag.spearman == null ? '–' : num(ag.spearman, 2)} />
        </div>
        <Table maxHeight={320} columns={[
          { key: 'label', label: 'variable (valeur de l\'individu)' },
          { key: 'shap', label: `SHAP (${w.space})`, num: true, render: (r) => num(r.shap, 3) },
          { key: 'lime', label: 'LIME (probabilité)', num: true, render: (r) => (r.lime == null ? 'hors top' : num(r.lime, 3)) },
          { key: 'same_sign', label: 'même sens ?', render: (r) => (r.same_sign == null ? '–' : r.same_sign ? '✓' : '✗') }]} rows={res.comparison} />
        <p className="muted">Les deux méthodes n'ont pas la même échelle ni la même construction : on compare surtout le signe et le classement. Un désaccord signale une zone où le modèle est instable ou fortement non linéaire.</p>
      </Card>
    </>
  )
}

// ---------------------------------------------------------------------------------------- équité
function Fairness({ model }) {
  const { threshold } = useApp()
  const [r, setR] = useState(null)
  const [st, run] = useAction()
  useEffect(() => { setR(null); run(async () => setR(await get(`/api/explain/sensitive/${model}?threshold=${threshold}`))) }, [model, threshold])

  return (
    <>
      <Notice kind="info">Détection automatique des variables potentiellement sensibles (sexe, âge, situation familiale, origine…) d'après leur nom. À vous de juger de la pertinence : une variable neutre en apparence peut servir de proxy.</Notice>
      <Status state={st} busy="Audit en cours…" />
      {r && r.features.length === 0 && <Notice kind="info">Aucune variable sensible détectée par nom dans ce jeu de données.</Notice>}
      {r && r.features.map((f) => (
        <Card key={f.feature} title={f.feature} right={<span className={'tag ' + (f.used_in_model ? 'warn' : '')}>{f.used_in_model ? 'utilisée par le modèle' : 'exclue du modèle (audit seul)'}</span>}>
          <div className="stats">
            <Stat label="part de l'impact SHAP" value={f.used_in_model ? pct(f.shap_share) : '–'} />
            <Stat label="écart de taux d'alerte entre groupes" value={f.parity_gap == null ? '–' : pct(f.parity_gap)} hint={`seuil ${r.threshold.toFixed(2)}`} />
          </div>
          <div className="grid2">
            <Chart height={260} data={[
              { type: 'bar', name: 'taux réel de positifs', x: f.groups.map((g) => g.group), y: f.groups.map((g) => g.actual_rate * 100), marker: { color: '#94a3b8' } },
              { type: 'bar', name: 'part signalée par le modèle', x: f.groups.map((g) => g.group), y: f.groups.map((g) => g.flagged_rate * 100), marker: { color: COLORS[0] } }]}
              layout={{ barmode: 'group', yaxis: { title: '%' } }} />
            <Table maxHeight={260} columns={[
              { key: 'group', label: 'groupe' }, { key: 'n', label: 'effectif', num: true },
              { key: 'flagged_rate', label: 'signalés', num: true, render: (g) => pct(g.flagged_rate) },
              { key: 'recall', label: 'rappel', num: true, render: (g) => (g.recall == null ? '–' : pct(g.recall)) }]} rows={f.groups} />
          </div>
        </Card>
      ))}
    </>
  )
}
