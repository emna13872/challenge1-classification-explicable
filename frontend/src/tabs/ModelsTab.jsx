import { useEffect, useState } from 'react'
import { useApp } from '../App.jsx'
import { get, post } from '../api.js'
import { Card, Table, Notice, useAction, Status, Stat, num } from '../ui.jsx'
import Chart, { COLORS } from '../Chart.jsx'
import FraudCard from './FraudCard.jsx'

const CAL_LABEL = { raw: 'brute (sans correction)', sigmoid: 'sigmoïde (Platt)', isotonic: 'isotonique' }

export default function ModelsTab() {
  const { prep, trained, setTrained, refreshEval, goTo, setThreshold, summary } = useApp()
  const [catalog, setCatalog] = useState(null)
  const [sel, setSel] = useState({})
  const [balance, setBalance] = useState(false)
  const [st, run] = useAction()
  const [cvSt, runCv] = useAction()
  const [cv, setCv] = useState(null)
  const [calSt, runCal] = useAction()
  const [cal, setCal] = useState(null)
  const [calModel, setCalModel] = useState('')
  const [thrSt, runThr] = useAction()
  const [thr, setThr] = useState(null)
  const [finSt, runFin] = useAction()
  const [fin, setFin] = useState(null)
  const [costFn, setCostFn] = useState(5)
  const [costFp, setCostFp] = useState(1)

  useEffect(() => {
    get('/api/models/catalog').then((c) => {
      setCatalog(c)
      const s = {}
      for (const [k, v] of Object.entries(c)) {
        s[k] = { on: true, tune: false, params: Object.fromEntries(Object.entries(v.params).map(([p, d]) => [p, d.default])) }
      }
      setSel(s)
    }).catch(() => {})
  }, [])

  if (!catalog) return <div className="busy"><span className="spin" /> Chargement du catalogue…</div>
  const upd = (k, patch) => setSel({ ...sel, [k]: { ...sel[k], ...patch } })
  const chosen = Object.entries(sel).filter(([, v]) => v.on)

  const train = () => run(async () => {
    const r = await post('/api/models/train', {
      balance,
      models: chosen.map(([name, v]) => ({ name, tune: v.tune, params: v.params })),
    })
    setTrained((prev) => {
      const map = new Map(prev.map((m) => [m.name, m]))
      r.trained.forEach((m) => map.set(m.name, m))
      return [...map.values()]
    })
    await refreshEval()
  })

  // Compare les modèles cochés par validation croisée répétée : le lot de test n'est JAMAIS utilisé ici.
  const compare = () => runCv(async () => {
    setCv(await post('/api/models/compare', {
      balance,
      models: chosen.map(([name, v]) => ({ name, params: v.params })),
    }))
  })

  // Mesure la fiabilité des probabilités du modèle choisi (lot d'apprentissage seul, jamais le test).
  const calName = chosen.some(([k]) => k === calModel) ? calModel : (chosen[0]?.[0] || '')
  const calibrate = () => runCal(async () => {
    setCal(await post('/api/models/calibration', {
      balance,
      model: { name: calName, params: sel[calName].params },
    }))
  })
  // Seuil de décision choisi par le coût des erreurs (lot d'apprentissage seul, jamais le test).
  const decide = () => runThr(async () => {
    setThr(await post('/api/models/threshold', {
      balance, cost_fn: Number(costFn), cost_fp: Number(costFp),
      model: { name: calName, params: sel[calName].params },
    }))
  })
  // Évaluation finale : le lot de test s'ouvre UNE fois ; un second clic renvoie le résultat déjà obtenu.
  const finalEval = (force = false) => runFin(async () => {
    setFin(await post('/api/final/evaluate', {
      balance, force, cost_fn: Number(costFn), cost_fp: Number(costFp),
      model: { name: calName, params: sel[calName].params },
    }))
  })
  const pc = (v) => (v == null ? '–' : (v * 100).toFixed(1) + ' %')
  const thrRows = thr ? [
    { k: 'Accepter tout le monde', cost: thr.accept_all, refuse: 0, recall: 0, precision: null },
    { k: 'Refuser tout le monde', cost: thr.refuse_all, refuse: 1, recall: 1, precision: thr.base_rate },
    { k: 'Seuil 50 % (par habitude)', cost: thr.at_half.cost, refuse: thr.at_half.refuse_rate, recall: thr.at_half.recall, precision: thr.at_half.precision },
    { k: `Seuil théorique ${num(thr.theoretical_threshold, 3)}`, cost: thr.at_theoretical.cost, refuse: thr.at_theoretical.refuse_rate, recall: thr.at_theoretical.recall, precision: thr.at_theoretical.precision },
    { k: `Meilleur seuil observé ${num(thr.best.threshold, 2)}`, cost: thr.best.cost, refuse: thr.best.refuse_rate, recall: thr.best.recall, precision: thr.best.precision },
  ] : []
  const calRows = cal ? Object.entries(cal.methods).map(([k, m]) => ({ key: k, ...m })) : []

  const cvRows = cv ? Object.entries(cv.models)
    .sort((a, b) => b[1].auc.mean - a[1].auc.mean)
    .map(([name, m]) => ({ name, ...m })) : []

  return (
    <div className="stack">
      <Card title="Choix des modèles et des hyperparamètres">
        <p className="muted">Cochez « Optimiser » pour lancer une recherche en grille (AUC-ROC, validation croisée à 3 plis) : les valeurs saisies sont alors remplacées par la meilleure combinaison de la grille.</p>
        <div className="models">
          {Object.entries(catalog).map(([k, m]) => (
            <div key={k} className={'model' + (sel[k]?.on ? ' on' : '')}>
              <label className="mh"><input type="checkbox" checked={!!sel[k]?.on} onChange={(e) => upd(k, { on: e.target.checked })} /> <b>{m.label}</b></label>
              <div className="params">
                {Object.entries(m.params).map(([p, d]) => (
                  <label key={p}>{p}
                    <input type="number" min={d.min} max={d.max} step={d.step} value={sel[k]?.params[p] ?? d.default} disabled={!sel[k]?.on || sel[k]?.tune}
                      onChange={(e) => upd(k, { params: { ...sel[k].params, [p]: e.target.value === '' ? '' : parseFloat(e.target.value) } })} />
                  </label>
                ))}
              </div>
              <label className="inline small"><input type="checkbox" disabled={!sel[k]?.on} checked={!!sel[k]?.tune} onChange={(e) => upd(k, { tune: e.target.checked })} /> Optimiser (grille : {Object.entries(m.grid).map(([g, v]) => `${g} ∈ {${v.join(', ')}}`).join(' ; ')})</label>
            </div>
          ))}
        </div>
        <label className="inline"><input type="checkbox" checked={balance} onChange={(e) => setBalance(e.target.checked)} /> Rééquilibrer les classes (poids de classe / scale_pos_weight) — utile si la classe positive est rare</label>
        <div className="row"><button className="primary" onClick={train} disabled={st.loading || chosen.length === 0}>Entraîner {chosen.length} modèle(s)</button></div>
        <Status state={st} busy="Entraînement en cours (l'optimisation peut prendre quelques dizaines de secondes)…" />
      </Card>

      <Card title="Quel modèle choisir ? Comparaison par validation croisée">
        <p className="muted">
          Chaque modèle coché est entraîné et noté plusieurs fois sur des découpages différents du lot d'apprentissage
          (validation croisée stratifiée répétée). On regarde la moyenne et l'écart-type de l'AUC. Un modèle « à égalité »
          est à moins d'un écart-type du meilleur : l'écart n'est pas assez grand pour être crédible.
          <b> Le lot de test n'est pas utilisé ici</b> : il reste intact pour la note finale.
        </p>
        <div className="row"><button onClick={compare} disabled={cvSt.loading || chosen.length < 2 || !prep}>Comparer {chosen.length} modèle(s)</button></div>
        <Status state={cvSt} busy="Validation croisée en cours (quelques dizaines de secondes sur un grand jeu)…" />
        {cv && (
          <>
            <Table maxHeight={300} columns={[
              { key: 'label', label: 'modèle' },
              { key: 'auc', label: 'AUC moyenne ± écart-type', render: (r) => `${num(r.auc.mean, 4)} ± ${num(r.auc.std, 4)}` },
              { key: 'pr', label: 'PR-AUC moyenne', num: true, render: (r) => num(r.pr_auc.mean, 4) },
              { key: 'verdict', label: 'verdict', render: (r) => (r.best ? 'meilleur' : r.tied_with_best ? 'à égalité avec le meilleur' : 'moins bon') },
              { key: 'seconds', label: 'temps (s)', num: true }]} rows={cvRows} />
            <p className="muted small">
              {cv.settings.n_splits} plis × {cv.settings.n_repeats} répétitions = {cv.settings.n_fits_per_model} notes par modèle,
              sur {cv.settings.n_rows_used} observations d'apprentissage. Test utilisé : {cv.test_set_used ? 'oui' : 'non'}.
            </p>
          </>
        )}
      </Card>

      <Card title="Les probabilités annoncées sont-elles fiables ? (calibration)">
        <p className="muted">
          Quand le modèle annonce « 30 % », environ 30 % des clients font-ils vraiment défaut ? On le mesure sur des
          prédictions faites par des modèles qui n'ont jamais vu le client noté (lot d'apprentissage seul, test intact),
          puis on essaie deux corrections. Le score de Brier est l'erreur moyenne au carré entre la probabilité annoncée
          et la réalité : plus il est petit, mieux c'est. Il se compare au Brier de référence d'un modèle qui annonce
          toujours le taux moyen.
        </p>
        <div className="row">
          <label className="inline">Modèle&nbsp;
            <select value={calName} onChange={(e) => setCalModel(e.target.value)} disabled={chosen.length === 0}>
              {chosen.map(([k]) => <option key={k} value={k}>{catalog[k].label}</option>)}
            </select>
          </label>
          <button onClick={calibrate} disabled={calSt.loading || !calName || !prep}>Mesurer la fiabilité</button>
        </div>
        <Status state={calSt} busy="Calcul des probabilités hors-pli…" />
        {cal && (
          <>
            <Notice kind={cal.recommended === 'raw' ? 'info' : 'warn'}>
              {cal.recommended === 'raw'
                ? 'Les probabilités brutes sont déjà fiables : aucune correction nécessaire.'
                : `Les probabilités brutes sont mal calibrées : la correction « ${CAL_LABEL[cal.recommended]} » les améliore nettement.`}
              {' '}Brier de référence : {num(cal.base_brier, 4)} (taux réel {num(cal.base_rate, 3)}).
            </Notice>
            <Table maxHeight={220} columns={[
              { key: 'm', label: 'version', render: (r) => CAL_LABEL[r.key] + (r.key === cal.recommended ? ' ← retenue' : '') },
              { key: 'brier', label: 'Brier', num: true, render: (r) => num(r.brier, 4) },
              { key: 'ece', label: 'écart annoncé / réel (ECE)', num: true, render: (r) => num(r.ece, 4) },
              { key: 'mp', label: 'proba moyenne annoncée', num: true, render: (r) => num(r.mean_pred, 3) }]} rows={calRows} />
            <Chart height={340} data={[
              { x: [0, 1], y: [0, 1], mode: 'lines', name: 'calibration parfaite', line: { dash: 'dot', color: '#9ca3af' } },
              ...calRows.map((r, i) => ({
                x: r.bins.map((b) => b.mean_pred), y: r.bins.map((b) => b.frac_pos), mode: 'lines+markers',
                name: CAL_LABEL[r.key], line: { color: COLORS[i] },
              })),
            ]} layout={{
              xaxis: { title: { text: 'probabilité annoncée' }, range: [0, 1] },
              yaxis: { title: { text: 'taux de défaut réel' }, range: [0, 1] },
            }} />
            <p className="muted small">
              Chaque point est un groupe de clients de même effectif. Plus la courbe est proche de la diagonale, plus les
              probabilités sont vraies. Test utilisé : {cal.test_set_used ? 'oui' : 'non'}.
            </p>
          </>
        )}
      </Card>

      <Card title="Quel seuil de décision ? (choix par le coût des erreurs)">
        <p className="muted">
          Deux erreurs ne coûtent pas le même prix : accorder un crédit à un client qui fait défaut (cas manqué), ou refuser
          un client qui aurait remboursé (fausse alerte). Indiquez ces deux coûts, dans la même unité (ce sont des
          hypothèses métier : à remplacer par les chiffres réels de la banque). Avec des probabilités fiables, on refuse
          dès que la probabilité dépasse <b>coût de la fausse alerte / (coût de la fausse alerte + coût du cas manqué)</b>.
        </p>
        <div className="row">
          <label className="inline">Modèle&nbsp;
            <select value={calName} onChange={(e) => setCalModel(e.target.value)} disabled={chosen.length === 0}>
              {chosen.map(([k]) => <option key={k} value={k}>{catalog[k].label}</option>)}
            </select>
          </label>
          <label className="inline">Coût d'un cas manqué&nbsp;
            <input type="number" min="0.01" step="0.5" value={costFn} onChange={(e) => setCostFn(e.target.value)} style={{ width: 80 }} />
          </label>
          <label className="inline">Coût d'une fausse alerte&nbsp;
            <input type="number" min="0.01" step="0.5" value={costFp} onChange={(e) => setCostFp(e.target.value)} style={{ width: 80 }} />
          </label>
          <button onClick={decide} disabled={thrSt.loading || !calName || !prep || !(costFn > 0) || !(costFp > 0)}>Chercher le meilleur seuil</button>
        </div>
        <Status state={thrSt} busy="Calcul du coût pour chaque seuil…" />
        {thr && (
          <>
            <Notice kind="info">
              Seuil retenu : <b>{num(thr.best.threshold, 2)}</b> (théorie : {num(thr.theoretical_threshold, 3)}).
              Coût moyen par dossier : {num(thr.best.cost, 3)} contre {num(thr.at_half.cost, 3)} avec le seuil de 50 %.
              Probabilités utilisées : {CAL_LABEL[thr.calibration_used]}.
            </Notice>
            <div className="row">
              <button onClick={() => setThreshold(Math.round(thr.best.threshold * 100) / 100)}>
                Utiliser le seuil {num(thr.best.threshold, 2)} dans les explications et la décision
              </button>
            </div>
            <Table maxHeight={260} columns={[
              { key: 'k', label: 'stratégie' },
              { key: 'cost', label: 'coût moyen par dossier', num: true, render: (r) => num(r.cost, 3) },
              { key: 'refuse', label: 'dossiers refusés', num: true, render: (r) => pc(r.refuse) },
              { key: 'recall', label: 'défauts repérés (rappel)', num: true, render: (r) => pc(r.recall) },
              { key: 'precision', label: 'refus justifiés (précision)', num: true, render: (r) => pc(r.precision) }]} rows={thrRows} />
            <Chart height={320} data={[
              { x: thr.curve.map((r) => r.threshold), y: thr.curve.map((r) => r.cost), mode: 'lines', name: 'coût moyen par dossier', line: { color: COLORS[0] } },
              { x: [thr.theoretical_threshold], y: [thr.at_theoretical.cost], mode: 'markers', name: 'seuil théorique', marker: { size: 11, color: COLORS[3] } },
              { x: [thr.best.threshold], y: [thr.best.cost], mode: 'markers', name: 'meilleur observé', marker: { size: 11, symbol: 'diamond', color: COLORS[2] } },
              { x: [0.5], y: [thr.at_half.cost], mode: 'markers', name: 'seuil 50 %', marker: { size: 11, symbol: 'x', color: '#dc2626' } },
            ]} layout={{
              xaxis: { title: { text: 'seuil : on refuse si la probabilité de défaut dépasse' }, range: [0, 1] },
              yaxis: { title: { text: 'coût moyen par dossier' } },
            }} />
            <p className="muted small">
              Calculé sur {thr.settings.n_rows_used} dossiers d'apprentissage, avec des probabilités hors-pli. Test utilisé : {thr.test_set_used ? 'oui' : 'non'}.
              Si « refuser tout le monde » coûte autant que le meilleur seuil, le modèle n'apporte rien avec ces coûts.
            </p>
          </>
        )}
      </Card>

      <Card title="Évaluation finale : ouvrir le lot de test une seule fois">
        <p className="muted">
          Étape finale de la chaîne. Le modèle, la correction de calibration et le seuil sont décidés <b>sans le lot de test</b>{' '}
          (cartes précédentes, mêmes coûts). Le modèle est ensuite entraîné sur tout l'apprentissage et le lot de test est lu
          <b> une seule fois</b>. On compare alors ce que la validation croisée promettait à ce qui se passe vraiment.
          Chaque ouverture est comptée : rouvrir le test pour changer d'avis en fait un lot de choix, et sa note devient trop optimiste.
        </p>
        <Notice kind="warn">
          L'onglet « Évaluation » affiche déjà des notes de test pour tous les modèles : ne vous en servez pas pour choisir un modèle.
        </Notice>
        <div className="row">
          <button className="primary" onClick={() => finalEval(false)} disabled={finSt.loading || !calName || !prep || !(costFn > 0) || !(costFp > 0)}>
            {fin ? 'Revoir le résultat final' : 'Ouvrir le lot de test (une seule fois)'}
          </button>
          {fin && <button onClick={() => finalEval(true)} disabled={finSt.loading}>Recalculer et rouvrir le test (compté)</button>}
        </div>
        <Status state={finSt} busy="Entraînement final puis lecture du lot de test…" />
        {fin && (
          <>
            <Notice kind={fin.openings > 1 ? 'warn' : 'info'}>
              Coût promis par la validation croisée : <b>{num(fin.expected.cost, 3)}</b> par dossier. Coût réel sur le lot de test :
              <b> {num(fin.test.at_threshold.cost, 3)}</b> (intervalle à 95 % : {num(fin.test.cost_ci95[0], 3)} à {num(fin.test.cost_ci95[1], 3)}).
              {fin.test.expected_inside_ci ? ' La promesse est tenue.' : ' La promesse sort de l\'intervalle : à examiner.'}
              {' '}Lot de test ouvert {fin.openings} fois{fin.already_opened ? ' (résultat déjà obtenu, le test n\'a pas été relu)' : ''}.
              {fin.warning ? ' ' + fin.warning : ''}
            </Notice>
            <Table maxHeight={260} columns={[
              { key: 'k', label: `sur les ${fin.test.n} dossiers du lot de test` },
              { key: 'cost', label: 'coût moyen par dossier', num: true, render: (r) => num(r.cost, 3) },
              { key: 'refuse', label: 'dossiers refusés', num: true, render: (r) => pc(r.refuse) },
              { key: 'recall', label: 'défauts repérés (rappel)', num: true, render: (r) => pc(r.recall) },
              { key: 'precision', label: 'refus justifiés (précision)', num: true, render: (r) => pc(r.precision) }]}
              rows={[
                { k: 'Accepter tout le monde', cost: fin.test.accept_all, refuse: 0, recall: 0, precision: null },
                { k: 'Refuser tout le monde', cost: fin.test.refuse_all, refuse: 1, recall: 1, precision: fin.test.base_rate },
                { k: 'Seuil 50 % (par habitude)', cost: fin.test.at_half.cost, refuse: fin.test.at_half.refuse_rate, recall: fin.test.at_half.recall, precision: fin.test.at_half.precision },
                { k: `Seuil retenu ${num(fin.threshold, 2)} (choisi sans le test)`, cost: fin.test.at_threshold.cost, refuse: fin.test.at_threshold.refuse_rate, recall: fin.test.at_threshold.recall, precision: fin.test.at_threshold.precision },
              ]} />
            <div className="stats">
              <Stat label="AUC sur le test" value={num(fin.test.auc, 3)} />
              <Stat label="Brier brut" value={num(fin.test.brier_raw, 4)} />
              <Stat label={`Brier corrigé (${CAL_LABEL[fin.calibration_used]})`} value={num(fin.test.brier_calibrated, 4)} />
              <Stat label="écart annoncé / réel (ECE), corrigé" value={num(fin.test.ece_calibrated, 4)} />
            </div>
            <Chart height={320} data={[
              { x: [0, 1], y: [0, 1], mode: 'lines', name: 'calibration parfaite', line: { dash: 'dot', color: '#9ca3af' } },
              { x: fin.test.bins_raw.map((b) => b.mean_pred), y: fin.test.bins_raw.map((b) => b.frac_pos), mode: 'lines+markers', name: 'brute', line: { color: COLORS[0] } },
              { x: fin.test.bins_calibrated.map((b) => b.mean_pred), y: fin.test.bins_calibrated.map((b) => b.frac_pos), mode: 'lines+markers', name: 'corrigée', line: { color: COLORS[2] } },
            ]} layout={{
              xaxis: { title: { text: 'probabilité annoncée' }, range: [0, 1] },
              yaxis: { title: { text: 'taux de défaut réel (lot de test)' }, range: [0, 1] },
            }} />
            <p className="muted small">
              Décisions prises sur {fin.settings.n_rows_used} dossiers d'apprentissage (probabilités hors-pli). Test utilisé : {fin.test_set_used ? 'oui, une lecture comptée' : 'non'}.
            </p>
          </>
        )}
      </Card>

      {prep && summary?.columns?.some((c) => c.name === 'Amount') && <FraudCard />}

      {trained.length > 0 && (
        <Card title="Modèles entraînés">
          <Table maxHeight={300} columns={[
            { key: 'label', label: 'modèle' },
            { key: 'params', label: 'hyperparamètres retenus', render: (r) => Object.entries(r.params || {}).map(([a, b]) => `${a}=${typeof b === 'number' ? +b.toPrecision(4) : b}`).join(', ') },
            { key: 'tuned', label: 'optimisé', render: (r) => (r.tuned ? 'oui' : 'non') },
            { key: 'cv_auc', label: 'AUC (validation croisée)', num: true, render: (r) => (r.cv_auc != null ? num(r.cv_auc) : '–') },
            { key: 'train_seconds', label: 'temps (s)', num: true }]} rows={trained} />
          <div className="right"><button className="primary" onClick={() => goTo('eval')}>Évaluer et comparer →</button></div>
        </Card>
      )}
      {!prep && <Notice kind="warn">Préparez les données avant d'entraîner.</Notice>}
    </div>
  )
}
