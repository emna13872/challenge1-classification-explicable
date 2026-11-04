import { useEffect, useState } from 'react'
import { useApp } from '../App.jsx'
import { post } from '../api.js'
import { Card, Stat, Notice, pct, useAction, Status, Slider } from '../ui.jsx'

export default function PrepTab() {
  const { summary: s, prep, setPrep, goTo } = useApp()
  const [feat, setFeat] = useState(new Set())
  const [cfg, setCfg] = useState({ num_impute: 'median', cat_impute: 'mode', scaling: 'standard', test_size: 0.25, random_state: 42, stratify: true, time_column: null })
  const [st, run] = useAction()
  // colonnes qui ressemblent à une date ou un instant : candidates pour un découpage dans le temps
  const timeCols = s.columns.filter((c) => c.kind === 'numeric' && c.name !== s.target && /time|date|temps|timestamp/i.test(c.name)).map((c) => c.name)

  useEffect(() => {
    // Fraude : par défaut, on apprend sur le passé et on teste sur le futur, et l'instant n'est pas une variable.
    const temporal = s.name === 'ulb_fraud' && timeCols.includes('Time') ? 'Time' : null
    setFeat(new Set(s.columns.filter((c) => c.name !== s.target && !c.suggest_exclude && c.name !== temporal).map((c) => c.name)))
    setCfg((c) => ({ ...c, time_column: temporal }))
  }, [s.name, s.target])

  const toggle = (n) => setFeat((f) => { const g = new Set(f); g.has(n) ? g.delete(n) : g.add(n); return g })
  const cols = s.columns.filter((c) => c.name !== s.target)
  const go = () => run(async () => setPrep(await post('/api/prep', { features: [...feat], ...cfg })))
  const set = (k) => (e) => setCfg({ ...cfg, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value })

  return (
    <div className="stack">
      <Card title="Variables explicatives" right={<>
        <button onClick={() => setFeat(new Set(cols.map((c) => c.name)))}>Tout cocher</button>{' '}
        <button onClick={() => setFeat(new Set())}>Tout décocher</button></>}>
        <p className="muted">Cible : <b>{s.target}</b>. Les identifiants (colonnes presque uniques) sont décochés par défaut. Retirez aussi toute variable connue seulement après l'événement (fuite de données).</p>
        <div className="checks">
          {cols.map((c) => (
            <label key={c.name} className={'check' + (c.suggest_exclude ? ' warn' : '')}>
              <input type="checkbox" checked={feat.has(c.name)} onChange={() => toggle(c.name)} />
              <span>{c.name}</span>
              <small>{c.kind === 'numeric' ? 'num.' : 'cat.'}{c.missing ? ` · ${c.missing} manq.` : ''}{c.suggest_exclude ? ' · identifiant ?' : ''}</small>
            </label>
          ))}
        </div>
      </Card>

      <Card title="Traitements">
        <div className="row wrap">
          <label>Valeurs manquantes numériques
            <select value={cfg.num_impute} onChange={set('num_impute')}>
              <option value="median">Médiane</option><option value="mean">Moyenne</option>
              <option value="zero">Zéro</option><option value="drop">Supprimer les lignes</option>
            </select></label>
          <label>Valeurs manquantes catégorielles
            <select value={cfg.cat_impute} onChange={set('cat_impute')}>
              <option value="mode">Modalité la plus fréquente</option><option value="drop">Supprimer les lignes</option>
            </select></label>
          <label>Normalisation
            <select value={cfg.scaling} onChange={set('scaling')}>
              <option value="standard">Standardisation (z-score)</option><option value="minmax">Min-max</option><option value="none">Aucune</option>
            </select></label>
          <label>Encodage catégoriel<input disabled value="One-hot" /></label>
        </div>
        <div className="row wrap">
          <Slider label="Part du jeu de test" value={cfg.test_size} min={0.1} max={0.5} step={0.05}
            onChange={(v) => setCfg({ ...cfg, test_size: v })} fmt={(v) => `${Math.round(v * 100)} %`} />
          <label>Graine aléatoire<input type="number" value={cfg.random_state} onChange={(e) => setCfg({ ...cfg, random_state: parseInt(e.target.value || '0', 10) })} /></label>
          <label>Découpage apprentissage / test
            <select value={cfg.time_column || ''} onChange={(e) => setCfg({ ...cfg, time_column: e.target.value || null })}>
              <option value="">au hasard</option>
              {timeCols.map((c) => <option key={c} value={c}>dans le temps (colonne {c}) : le test = les plus récents</option>)}
            </select></label>
          <label className="inline"><input type="checkbox" checked={cfg.stratify} disabled={!!cfg.time_column} onChange={set('stratify')} /> Découpage stratifié</label>
        </div>
        {cfg.time_column && (
          <Notice kind="info">
            Découpage dans le temps : on apprend sur les {Math.round((1 - cfg.test_size) * 100)} % de lignes les plus anciennes et on teste sur
            les plus récentes, comme un modèle mis en service qui ne voit que le passé.
            {feat.has(cfg.time_column) && ` Attention : « ${cfg.time_column} » est aussi cochée comme variable ; le test contient des instants jamais vus à l'apprentissage, mieux vaut la décocher.`}
          </Notice>
        )}
        <div className="row">
          <button className="primary" disabled={st.loading || feat.size === 0} onClick={go}>Préparer les données ({feat.size} variables)</button>
        </div>
        <Status state={st} />
      </Card>

      {prep && (
        <Card title="Résultat de la préparation">
          <div className="stats">
            <Stat label="entraînement" value={prep.n_train.toLocaleString('fr-FR')} hint={`classe positive : ${pct(prep.train_pos_rate)}`} />
            <Stat label="test" value={prep.n_test.toLocaleString('fr-FR')} hint={`classe positive : ${pct(prep.test_pos_rate)}`} />
            <Stat label="variables d'entrée" value={prep.n_features_in} />
            <Stat label="colonnes après encodage" value={prep.n_features_encoded} />
          </div>
          {prep.dropped_rows > 0 && <Notice kind="warn">{prep.dropped_rows} lignes supprimées (valeurs manquantes).</Notice>}
          {prep.split === 'temporal' && <p className="muted">Découpage dans le temps (colonne {prep.time_column}) : tout le test est postérieur à tout l'apprentissage.</p>}
          <p className="muted">Imputation et normalisation sont ajustées sur l'entraînement uniquement (pas de fuite vers le test).</p>
          <div className="right"><button className="primary" onClick={() => goTo('models')}>Passer à la modélisation →</button></div>
        </Card>
      )}
    </div>
  )
}
