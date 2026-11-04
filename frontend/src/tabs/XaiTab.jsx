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
            {[['global', 'Explications globales'], ['local', 'Explication d\'un individu'], ['fair', 'Variables sensibles et équité'], ['audit', 'Audit du recours']].map(([k, t]) => (
              <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{t}</button>))}
          </div>
        </div>
      </Card>
      {view === 'global' && <Global model={cur} label={label} names={names} />}
      {view === 'local' && <Local model={cur} />}
      {view === 'fair' && <Fairness model={cur} />}
      {view === 'audit' && <RecourseAudit model={cur} />}
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
  const [dec, setDec] = useState(null)
  const [decSt, runDec] = useAction()

  useEffect(() => {
    get(`/api/explain/test-samples/${model}?limit=40`).then((s) => { setSamples(s); if (s[0]) setIdx(s[0].index) }).catch(() => {})
    get('/api/explain/form').then((f) => { setForm(f); setRec(Object.fromEntries(f.fields.map((x) => [x.name, x.default]))) }).catch(() => {})
    setRes(null)
    setDec(null)
  }, [model])

  const explain = () => run(async () => {
    const body = { model, threshold, ...(mode === 'test' ? { test_index: Number(idx) } : { record: rec }) }
    setRes(await post('/api/explain/local', body))
  })

  // Décision, motifs de refus et recours : deux appels à l'API, l'un après l'autre (même client, même seuil).
  const decide = () => runDec(async () => {
    const who = mode === 'test' ? { test_index: Number(idx) } : { record: rec }
    const thr = Math.min(0.99, Math.max(0.01, threshold))
    const reasons = await post('/api/decision/reasons', { model, threshold: thr, ...who })
    const recourse = await post('/api/decision/recourse', { model, threshold: thr, ...who })
    setDec({ reasons, recourse })
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
          <button onClick={decide} disabled={decSt.loading}>Motifs de la décision et conseil</button>
        </div>
        <Status state={st} busy="Calcul de SHAP et LIME…" />
        <Status state={decSt} busy="Calcul des motifs et recherche d'un conseil réaliste…" />
      </Card>
      {dec && <DecisionResult d={dec} />}
      {res && <LocalResult res={res} />}
    </>
  )
}

const RECOURSE_NOTE = {
  already_accepted: 'Ce client est déjà accepté au seuil choisi : aucun changement nécessaire.',
  no_actionable: "Aucune variable modifiable n'est déclarée pour ce jeu de données : l'application ne propose pas de conseil plutôt que d'en inventer un.",
}

function DecisionResult({ d }) {
  const r = d.reasons
  const c = d.recourse
  const refused = r.decision === 'refus'
  return (
    <>
      <Card title="Décision et motifs">
        <div className="stats">
          <Stat label={`décision au seuil ${pct(r.threshold, 1)}`} value={refused ? 'REFUS' : 'ACCORD'} />
          <Stat label="risque annoncé par le modèle" value={pct(r.probability, 1)} />
        </div>
        {r.reasons_against.length > 0 && (
          <>
            <h4>Motifs principaux qui poussent vers le refus</h4>
            <Table maxHeight={220} columns={[
              { key: 'text', label: 'motif' },
              { key: 'share', label: 'part de la poussée', num: true, render: (x) => pct(x.share_of_push, 0) }]} rows={r.reasons_against} />
          </>
        )}
        {r.reasons_for.length > 0 && (
          <>
            <h4>Ce qui joue en sa faveur</h4>
            <Table maxHeight={180} columns={[
              { key: 'text', label: 'élément' },
              { key: 'share', label: 'part', num: true, render: (x) => pct(x.share_of_push, 0) }]} rows={r.reasons_for} />
          </>
        )}
        {r.excluded_sensitive.length > 0 && (
          <Notice kind="info">
            Variables sensibles jamais citées comme motif : {r.excluded_sensitive.join(', ')}. Elles sont présentes dans le modèle
            et font l'objet de l'audit d'équité (vue « Variables sensibles et équité »).
          </Notice>
        )}
        <p className="muted small">
          Ces motifs décrivent ce que le modèle a appris (des corrélations), pas des causes. Les contributions s'additionnent
          jusqu'à la prédiction (écart de contrôle : {r.additivity_gap.toExponential(1)}).
        </p>
      </Card>

      <Card title="Que changer pour être accepté ?">
        {RECOURSE_NOTE[c.status] && <Notice kind="info">{RECOURSE_NOTE[c.status]}</Notice>}
        {c.status === 'found' && (
          <ol className="options">
            {c.options.map((o, i) => <li key={i}>{o.text}</li>)}
          </ol>
        )}
        {c.status === 'none_found' && c.mode === 'future' && (
          <Notice kind="warn">
            Aucun conseil réaliste trouvé, même pour une nouvelle demande dans {c.horizons.join(', ')} mois en payant sans aucun retard
            (risque le plus bas atteignable : {pct(c.lowest_probability_found, 1)}, seuil {pct(c.threshold, 1)}).
            {c.assumed_bill <= 0 ? " Ce client n'a pas de facture en cours : le modèle n'a pas de remboursement à observer, et on ne conseille jamais de dépenser pour paraître fiable." : " Ce qui pèse ici ne se change pas (limite fixée par la banque, historique qui reste dans la fenêtre…)."}
          </Notice>
        )}
        {c.status === 'none_found' && c.mode !== 'future' && (
          <Notice kind="warn">
            Aucun conseil réaliste trouvé. En changeant au mieux tout ce qui est modifiable
            ({Object.keys(c.actionable).join(', ')}), le risque annoncé ne descend pas sous {pct(c.lowest_probability_found, 1)},
            alors que le seuil est de {pct(c.threshold, 1)}. Ce qui pèse le plus ici ne se change pas (historique passé, limite fixée par la banque…).
          </Notice>
        )}
        {c.mode === 'future' && c.assumptions?.length > 0 && (
          <div className="muted small">
            Le passé (retards, factures, montants déjà remboursés) ne se réécrit pas : le conseil porte sur les mois à venir,
            pour une nouvelle demande. Hypothèses :
            <ul>{c.assumptions.map((h, i) => <li key={i}>{h}</li>)}</ul>
          </div>
        )}
        {c.mode !== 'future' && Object.keys(c.actionable).length > 0 && (
          <p className="muted small">Variables que le client peut changer : {Object.keys(c.actionable).join(', ')}. Les autres sont considérées comme fixes.</p>
        )}
        <p className="muted small">{c.caveat}</p>
      </Card>
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

// ---------------------------------------------------------------------------------------- audit du recours (B4)
const ci = (c) => (c ? `entre ${pct(c[0], 0)} et ${pct(c[1], 0)}` : '')

function RecourseAudit({ model }) {
  const { threshold } = useApp()
  const [size, setSize] = useState(200)
  const [a, setA] = useState(null)
  const [st, run] = useAction()
  useEffect(() => setA(null), [model, threshold])
  const thr = Math.min(0.99, Math.max(0.01, threshold))
  const go = () => run(async () => setA(await post('/api/decision/audit', { model, threshold: thr, max_clients: size || null })))

  return (
    <>
      <Card title="Audit du recours : tous les clients refusés reçoivent-ils un conseil ?">
        <p className="muted small">
          Un conseil trouvé pour un client ne dit rien des autres. Ici, on cherche un conseil pour chaque client refusé
          du lot de test (seuil {pct(thr, 1)}), puis on compte : combien en reçoivent un, quel effort il demande,
          et si l'accès au conseil est le même d'un groupe à l'autre (sexe, âge…).
        </p>
        <div className="row wrap">
          <label>Clients audités
            <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
              <option value={100}>100 au hasard (rapide)</option>
              <option value={200}>200 au hasard</option>
              <option value={500}>500 au hasard</option>
              <option value={0}>tous (environ 40 s sur Taïwan)</option>
            </select></label>
          <button className="primary" onClick={go} disabled={st.loading}>Lancer l'audit</button>
        </div>
        <Status state={st} busy="Recherche d'un conseil pour chaque client refusé…" />
      </Card>
      {a && <AuditResult a={a} />}
    </>
  )
}

function AuditResult({ a }) {
  if (a.status === 'no_refused') return <Notice kind="info">Aucun client refusé à ce seuil : rien à auditer.</Notice>
  if (a.status === 'no_actionable') return <Notice kind="info">Aucune variable modifiable n'est déclarée pour ce jeu de données : pas de conseil, donc pas d'audit.</Notice>
  const e = a.effort
  const vars = Object.entries(e.variables_used || {})
  return (
    <>
      <Card title="Combien de clients refusés reçoivent un conseil ?">
        <div className="stats">
          <Stat label="clients refusés (lot de test)" value={`${a.n_refused} / ${a.n_test}`} hint={pct(a.share_refused, 0)} />
          <Stat label={a.sampled ? `audités (tirés au hasard)` : 'audités (tous)'} value={a.n_audited} hint={`${a.seconds} s`} />
          <Stat label="reçoivent un conseil réaliste" value={pct(a.share_found, 0)} hint={a.sampled ? ci(a.share_found_ci95) : 'mesure exacte'} />
          <Stat label="aucun conseil possible" value={pct(a.share_none, 0)} />
        </div>
        {a.sampled && (
          <p className="muted small">
            Échantillon : le vrai pourcentage sur tous les refusés est très probablement {ci(a.share_found_ci95)} (intervalle de confiance à 95 %).
          </p>
        )}
      </Card>

      <Card title="Quel effort demandent les conseils ?">
        {e.horizon_months && (
          <div className="stats">
            {Object.entries(e.horizon_months).map(([h, k]) => (
              <Stat key={h} label={`conseil possible dès ${h} mois`} value={pct(k / a.n_found, 0)} hint={`${k} clients`} />))}
            <Stat label="payer la facture en entier" value={pct(e.share_pays_full_bill, 0)} />
          </div>
        )}
        <div className="stats">
          {!e.horizon_months && <Stat label="une seule chose à changer" value={pct(e.share_one_change, 0)} />}
          <Stat label="saut médian (en centiles)" value={e.median_percentile_jump == null ? '–' : `${Math.round(e.median_percentile_jump * 100)} places`} hint={`90 % des sauts ≤ ${Math.round((e.p90_percentile_jump || 0) * 100)} places`} />
          <Stat label="montant multiplié par (médiane)" value={e.median_ratio == null ? '–' : `× ${e.median_ratio.toFixed(1)}`} />
          <Stat label="partent de zéro" value={pct(e.share_from_zero, 0)} />
        </div>
        {vars.length > 1 && (
          <Chart height={240} data={[{ type: 'bar', x: vars.map((v) => v[0]), y: vars.map((v) => v[1]), marker: { color: COLORS[0] } }]}
            layout={{ yaxis: { title: 'nombre de conseils' } }} />
        )}
        <p className="muted small">
          Un saut de 10 places de centile veut dire : passer, par exemple, du 40e au 50e rang sur 100 clients classés par montant.
          « Partent de zéro » : le client ne remboursait rien jusqu'ici, le conseil lui demande de commencer.
        </p>
      </Card>

      {a.n_none > 0 && (
        <Card title="Pourquoi certains n'ont aucun conseil ?">
          <div className="stats">
            <Stat label="risque médian : sans conseil" value={pct(a.none_found.median_probability, 0)} />
            <Stat label="risque médian : avec conseil" value={pct(a.none_found.median_probability_when_found, 0)} />
            <Stat label="risque le plus bas atteignable (médiane, sans conseil)" value={pct(a.none_found.median_lowest_probability, 0)} hint={`seuil ${pct(a.threshold, 1)}`} />
            {a.none_found.share_no_bill != null && <Stat label="sans facture en cours" value={pct(a.none_found.share_no_bill, 0)} hint="carte inutilisée le dernier mois" />}
          </div>
          <p className="muted small">
            {a.none_found.share_no_bill != null
              ? "En mode « avenir », le conseil ne demande jamais de rembourser plus que la facture. Les clients sans conseil sont surtout ceux dont la carte est peu ou pas utilisée : il n'y a rien à rembourser, et on ne conseille jamais de dépenser pour paraître fiable."
              : "Même en changeant au mieux ce qui est modifiable, ces clients restent au-dessus du seuil : ce qui pèse chez eux ne se change pas."}
          </p>
        </Card>
      )}

      <Card title="L'accès au conseil est-il le même pour tous ?">
        {a.fairness.length === 0 && <Notice kind="info">Aucune variable sensible avec des groupes assez grands (au moins 30 clients) pour comparer.</Notice>}
        {a.fairness.map((f) => (
          <div key={f.variable}>
            <h4>{f.variable} · écart max {pct(f.gap, 1)}</h4>
            {f.alert && <Notice kind="warn">Écart d'au moins 10 points entre groupes : à examiner (variable fixe corrélée au groupe ?).</Notice>}
            <Table maxHeight={200} columns={[
              { key: 'group', label: 'groupe' }, { key: 'n', label: 'refusés audités', num: true },
              { key: 'share_found', label: 'reçoivent un conseil', num: true, render: (g) => pct(g.share_found, 0) },
              { key: 'ci95', label: 'intervalle à 95 %', render: (g) => ci(g.ci95) }]} rows={f.groups} />
          </div>
        ))}
        <p className="muted small">
          Les variables sensibles ne sont jamais des motifs ni des conseils, mais d'autres variables peuvent leur être liées :
          seule cette mesure par groupe le révèle. {a.caveat}
        </p>
      </Card>
    </>
  )
}
