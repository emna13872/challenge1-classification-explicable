import { useEffect, useState } from 'react'
import { useApp } from '../App.jsx'
import { get, post } from '../api.js'
import { Card, Table, Notice, useAction, Status, num } from '../ui.jsx'

export default function ModelsTab() {
  const { prep, trained, setTrained, refreshEval, goTo } = useApp()
  const [catalog, setCatalog] = useState(null)
  const [sel, setSel] = useState({})
  const [balance, setBalance] = useState(false)
  const [st, run] = useAction()

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
