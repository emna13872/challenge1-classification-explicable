// Client minimal de l'API FastAPI. Chaque onglet de navigateur a sa propre session serveur.
const BASE = import.meta.env.VITE_API_URL || ''

let sid = null
try { sid = sessionStorage.getItem('sid') } catch { /* stockage indisponible */ }
if (!sid) {
  sid = (globalThis.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2)
  try { sessionStorage.setItem('sid', sid) } catch { /* ignore */ }
}

export const apiBase = BASE || window.location.origin
export const sessionId = sid

async function request(path, opts = {}) {
  const isForm = opts.body instanceof FormData
  const res = await fetch(BASE + path, {
    ...opts,
    headers: { 'X-Session-Id': sid, ...(opts.body && !isForm ? { 'Content-Type': 'application/json' } : {}) },
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch { /* corps non JSON */ }
    throw new Error(msg)
  }
  return res.json()
}

export const get = (p) => request(p)
export const post = (p, body) => request(p, { method: 'POST', body: JSON.stringify(body) })
export const upload = (p, formData) => request(p, { method: 'POST', body: formData })
