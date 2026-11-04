import { useEffect, useState } from 'react'
import { useApp } from '../App.jsx'
import { get, post } from '../api.js'
import { Card, Notice, Table, useAction, Status } from '../ui.jsx'

// Phase D : l'assistant (un LLM local via Ollama) répond en appelant les outils ; chaque nombre est vérifié.
const STATUS = {
  'vérifié': ['ok', 'Vérifié : chaque nombre vient d\'un outil'],
  'corrigé': ['ok', 'Corrigé : un nombre inventé a été repris, la réponse finale est vérifiée'],
  'non vérifié': ['bad', 'Non vérifié : des nombres ne viennent d\'aucun outil (surlignés)'],
  'sans outil': ['bad', 'Sans outil : aucun outil n\'a vraiment été appelé (écrire « je vais appeler l\'outil » ne suffit pas), la réponse ne s\'appuie sur aucune donnée'],
  'sans réponse': ['bad', 'Sans réponse : l\'assistant n\'a rien écrit'],
  'interrompu': ['bad', 'Interrompu : trop d\'appels d\'outils sans réponse'],
}

function suggestions(dataset) {
  const s = [
    'Pourquoi le client n°12 est-il refusé ?',
    'Que peut faire le client n°12 pour être accepté ?',
    'Le client n°12 est-il refusé à cause de son âge ?',
    'L\'accès au conseil est-il le même pour les femmes et les hommes ?',
    'Qu\'est-ce que le seuil de décision ?',
  ]
  return dataset === 'ulb_fraud'
    ? ['Faut-il vérifier la transaction n°12 ?', 'Combien de pertes la détection de fraude évite-t-elle ?', 'Que doit faire le titulaire de la transaction n°12 pour éviter une alerte ?', 'Qu\'est-ce que la règle du montant ?']
    : s
}

function Marked({ text, bad }) {
  if (!bad.length) return <>{text}</>
  const esc = bad.map((b) => b.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const parts = text.split(new RegExp(`(${esc.join('|')})`, 'g'))
  return <>{parts.map((p, i) => (bad.includes(p) ? <mark key={i} title="nombre sans source">{p}</mark> : <span key={i}>{p}</span>))}</>
}

export default function AgentView({ model }) {
  const { threshold, summary } = useApp()
  const [status, setStatus] = useState(null)
  const [llm, setLlm] = useState('')
  const [q, setQ] = useState('Pourquoi le client n°12 est-il refusé ?')
  const [mode, setMode] = useState('guided')
  const [res, setRes] = useState(null)
  const [st, run] = useAction()
  const [stSt, runSt] = useAction()

  const check = () => runSt(async () => {
    const s = await get('/api/agent/status')
    setStatus(s)
    if (s.default) setLlm((cur) => cur || s.default)
  })
  useEffect(() => { check() }, [])

  const ask = () => run(async () => {
    setRes(null)
    setRes(await post('/api/agent/ask', { question: q, model, mode, threshold: Math.min(0.99, Math.max(0.01, threshold)), llm_model: llm || null }))
  })
  const chatModels = (status?.models || []).filter((m) => !m.name.includes('embed'))
  const [kind, label] = res ? (STATUS[res.status] || ['bad', res.status]) : []

  return (
    <>
      <Card title="Assistant : poser une question sur les décisions">
        <p className="muted">
          Un LLM (grand modèle de langage) tourne sur ton PC avec Ollama. Il n'a pas le droit d'inventer : il appelle les outils
          de CreditLens (décision, motifs, conseil, audit, fraude, définitions) et rédige à partir de leurs résultats.
          Un garde-fou vérifie ensuite chaque nombre de sa réponse.
        </p>
        <Status state={stSt} busy="Recherche d'Ollama…" />
        {status && !status.ok && (
          <Notice kind="warn">
            {status.message} <button onClick={check}>Réessayer</button>
          </Notice>
        )}
        {status?.ok && (
          <div className="row wrap">
            <label>Modèle Ollama
              <select value={llm} onChange={(e) => setLlm(e.target.value)}>
                {chatModels.map((m) => <option key={m.name} value={m.name}>{m.name}{m.parameters ? ` · ${m.parameters}` : ''}{m.size_gb ? ` · ${m.size_gb} Go` : ''}</option>)}
              </select></label>
            <label>Qui choisit les outils ?
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="guided">l'application (mode guidé, conseillé pour un petit modèle)</option>
                <option value="agent">le LLM lui-même (mode agent)</option>
              </select></label>
            <span className="muted small">Modèle expliqué : {model} · seuil {Math.round(threshold * 1000) / 10} %</span>
          </div>
        )}
        {status?.ok && (
          <p className="muted small">
            Mode guidé : l'application lit la question (n° de client, équité, fraude, seuil…), appelle elle-même les outils, et le LLM
            ne fait que rédiger. Mode agent : le LLM décide quels outils appeler ; il faut un modèle assez grand pour le faire de façon fiable.
          </p>
        )}
        <div className="row wrap">
          {suggestions(summary?.name).map((s) => <button key={s} className="small" onClick={() => setQ(s)}>{s}</button>)}
        </div>
        <textarea rows={3} value={q} onChange={(e) => setQ(e.target.value)} style={{ width: '100%' }} />
        <div className="row">
          <button className="primary" onClick={ask} disabled={st.loading || !status?.ok || q.trim().length < 3}>Demander</button>
        </div>
        <Status state={st} busy="L'assistant appelle les outils puis rédige… (sur un PC sans carte graphique, compter une à trois minutes)" />
      </Card>

      {res && (
        <Card title="Réponse" right={<span className={`tag ${kind}`}>{res.status}</span>}>
          <div className={`answer ${kind}`}><Marked text={res.answer || '(vide)'} bad={res.verification?.unsupported || []} /></div>
          <p className="small"><b>{label}.</b> {res.verification ? `${res.verification.n_supported} nombre(s) sur ${res.verification.n_numbers} appuyé(s) par un outil ou par la question.` : ''}
            {res.corrections ? ' Une correction a été demandée.' : ''}{res.nudges ? ' L\'assistant a dû être relancé pour appeler un outil.' : ''}</p>
          {res.verification?.warnings?.map((w) => <Notice key={w} kind="warn">{w}</Notice>)}
          {res.verification?.numbers?.length > 0 && (
            <Table maxHeight={200} columns={[
              { key: 'text', label: 'nombre dans la réponse' },
              { key: 'source', label: 'd\'où il vient', render: (r) => r.source || 'aucune source' },
              { key: 'supported', label: 'appuyé ?', render: (r) => (r.supported ? 'oui' : 'NON') }]} rows={res.verification.numbers} />
          )}
          <details>
            <summary>Appels d'outils ({res.trace.length})</summary>
            {res.trace.map((t, i) => (
              <div key={i}>
                <p className="small"><b>{i + 1}. {t.tool}</b>({JSON.stringify(t.arguments)}){t.by === 'application' ? ' · choisi par l\'application' : ' · choisi par le LLM'}</p>
                <pre style={{ maxHeight: 220 }}>{JSON.stringify(t.result, null, 2)}</pre>
              </div>
            ))}
          </details>
          <p className="muted small">{res.llm.provider} · {res.llm.model} · mode {res.mode} · {res.seconds} s au total</p>
        </Card>
      )}
    </>
  )
}
