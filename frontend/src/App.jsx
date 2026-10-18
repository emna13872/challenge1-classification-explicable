import { createContext, useContext, useState, useCallback } from 'react'
import { get, sessionId } from './api.js'
import DataTab from './tabs/DataTab.jsx'
import PrepTab from './tabs/PrepTab.jsx'
import ModelsTab from './tabs/ModelsTab.jsx'
import EvalTab from './tabs/EvalTab.jsx'
import XaiTab from './tabs/XaiTab.jsx'
import ApiTab from './tabs/ApiTab.jsx'

const AppCtx = createContext(null)
export const useApp = () => useContext(AppCtx)

const TABS = [
  ['data', '1 · Données', DataTab],
  ['prep', '2 · Préparation', PrepTab],
  ['models', '3 · Modèles', ModelsTab],
  ['eval', '4 · Évaluation', EvalTab],
  ['xai', '5 · Explications', XaiTab],
  ['api', '6 · API', ApiTab],
]

export default function App() {
  const [tab, setTab] = useState('data')
  const [summary, setSummaryRaw] = useState(null)
  const [prep, setPrep] = useState(null)
  const [trained, setTrained] = useState([])
  const [evalData, setEvalData] = useState(null)
  const [threshold, setThreshold] = useState(0.5)

  // Un nouveau jeu de données ou une nouvelle cible invalide tout l'aval.
  const setSummary = useCallback((s) => {
    setSummaryRaw(s); setPrep(null); setTrained([]); setEvalData(null)
  }, [])
  const refreshEval = useCallback(async () => {
    setEvalData(await get('/api/evaluate?threshold=0.5'))
  }, [])

  const ctx = { summary, setSummary, prep, setPrep: (p) => { setPrep(p); setTrained([]); setEvalData(null) },
    trained, setTrained, evalData, refreshEval, threshold, setThreshold, goTo: setTab }
  const Active = TABS.find((t) => t[0] === tab)[2]
  const ready = { data: true, prep: !!summary, models: !!prep, eval: trained.length > 0, xai: trained.length > 0, api: true }

  return (
    <AppCtx.Provider value={ctx}>
      <div className="app">
        <header className="top">
          <div>
            <h1>Classification explicable</h1>
            <p>Data Mining · II3-IF · ENSI 2026-2027 — {summary ? <>jeu : <b>{summary.name}</b> · cible : <b>{summary.target} = {summary.positive}</b></> : 'aucun jeu chargé'}</p>
          </div>
          <code className="sid" title="Identifiant de session envoyé à l'API">session {sessionId.slice(0, 8)}</code>
        </header>
        <nav className="tabs">
          {TABS.map(([k, label]) => (
            <button key={k} className={(tab === k ? 'on ' : '') + (ready[k] ? '' : 'off')} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
        <main>
          {ready[tab] ? <Active /> : <div className="notice info">Terminez d'abord l'étape précédente.</div>}
        </main>
      </div>
    </AppCtx.Provider>
  )
}
