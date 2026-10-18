import { useState } from 'react'

export const Card = ({ title, right, children, className = '' }) => (
  <section className={'card ' + className}>
    {(title || right) && <header><h3>{title}</h3><div>{right}</div></header>}
    {children}
  </section>
)

export const Stat = ({ label, value, hint }) => (
  <div className="stat"><div className="stat-v">{value}</div><div className="stat-l">{label}</div>{hint && <div className="stat-h">{hint}</div>}</div>
)

export const Notice = ({ kind = 'info', children }) => <div className={'notice ' + kind}>{children}</div>

export const pct = (v, d = 1) => (v == null ? '–' : (v * 100).toFixed(d) + ' %')
export const num = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d))

export function Table({ columns, rows, maxHeight = 360, dense = true }) {
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table className={dense ? 'dense' : ''}>
        <thead><tr>{columns.map((c) => <th key={c.key} className={c.num ? 'r' : ''}>{c.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={r._cls || ''}>
              {columns.map((c) => {
                const v = c.render ? c.render(r) : r[c.key]
                return <td key={c.key} className={c.num ? 'r' : ''}>{v == null ? '' : String(v)}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Exécute une action asynchrone en suivant l'état de chargement et l'erreur.
export function useAction() {
  const [state, setState] = useState({ loading: false, error: null })
  const run = async (fn) => {
    setState({ loading: true, error: null })
    try {
      const r = await fn()
      setState({ loading: false, error: null })
      return r
    } catch (e) {
      setState({ loading: false, error: e.message || String(e) })
      return null
    }
  }
  return [state, run]
}

export const Status = ({ state, busy = 'Calcul en cours…' }) => (
  <>
    {state.loading && <div className="busy"><span className="spin" /> {busy}</div>}
    {state.error && <Notice kind="error">{state.error}</Notice>}
  </>
)

export const Slider = ({ label, value, min, max, step, onChange, fmt = (v) => v }) => (
  <label className="slider">
    <span>{label} <b>{fmt(value)}</b></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
  </label>
)
