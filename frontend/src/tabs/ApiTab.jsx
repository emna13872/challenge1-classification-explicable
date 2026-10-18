import { useState } from 'react'
import { useApp } from '../App.jsx'
import { get, apiBase, sessionId } from '../api.js'
import { Card, Table, useAction, Status } from '../ui.jsx'

const ENDPOINTS = [
  ['GET', '/api/health', "État du service"],
  ['GET', '/api/datasets', 'Jeux de données téléchargeables automatiquement'],
  ['POST', '/api/data/load', 'Télécharge un jeu public (UCI / GitHub / OpenML)'],
  ['POST', '/api/data/upload', 'Importe un CSV et sa variable cible'],
  ['POST', '/api/data/target', 'Choisit la cible et la classe positive'],
  ['GET', '/api/data/column/{col}', "Distribution et taux de la classe positive d'une variable"],
  ['GET', '/api/data/correlation', 'Matrice de corrélation'],
  ['POST', '/api/prep', 'Imputation, encodage, normalisation, découpage train/test'],
  ['GET', '/api/models/catalog', 'Modèles, hyperparamètres et grilles'],
  ['POST', '/api/models/train', 'Entraîne et optimise les modèles choisis'],
  ['GET', '/api/evaluate', 'Métriques, courbes ROC / PR, table de seuils'],
  ['GET', '/api/explain/global/{model}', 'Importance native et SHAP globaux (beeswarm, dépendance)'],
  ['GET', '/api/explain/permutation', 'Importance par permutation, comparée entre modèles'],
  ['POST', '/api/explain/local', 'Explication SHAP + LIME + phrase de synthèse pour un individu'],
  ['GET', '/api/explain/sensitive/{model}', 'Audit des variables sensibles et écarts entre groupes'],
]

export default function ApiTab() {
  const { threshold } = useApp()
  const [health, setHealth] = useState(null)
  const [st, run] = useAction()
  const curl = `curl -X POST ${apiBase}/api/explain/local \\
  -H "Content-Type: application/json" \\
  -H "X-Session-Id: ${sessionId}" \\
  -d '{"model": "xgb", "test_index": 3, "threshold": ${threshold.toFixed(2)}}'`

  return (
    <div className="stack">
      <Card title="API d'interprétation (FastAPI)">
        <p>Toute l'application passe par cette API : l'interface React ne calcule rien d'autre que l'affichage. La documentation interactive (Swagger) est générée automatiquement.</p>
        <div className="row wrap">
          <a className="btn" href={apiBase + '/docs'} target="_blank" rel="noreferrer">Ouvrir /docs (Swagger)</a>
          <a className="btn" href={apiBase + '/redoc'} target="_blank" rel="noreferrer">Ouvrir /redoc</a>
          <button onClick={() => run(async () => setHealth(await get('/api/health')))}>Tester /api/health</button>
          {health && <code>{JSON.stringify(health)}</code>}
        </div>
        <Status state={st} />
        <p className="muted">Les données et modèles sont conservés par session (en-tête <code>X-Session-Id</code>).</p>
      </Card>
      <Card title="Exemple d'appel (session courante)"><pre>{curl}</pre></Card>
      <Card title="Routes principales">
        <Table maxHeight={480} columns={[{ key: 0, label: 'méthode' }, { key: 1, label: 'route' }, { key: 2, label: 'rôle' }]}
          rows={ENDPOINTS.map((e) => ({ 0: e[0], 1: e[1], 2: e[2] }))} />
      </Card>
    </div>
  )
}
