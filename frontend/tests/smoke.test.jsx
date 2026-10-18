// Test de fumée de l'interface : parcourt les 6 modules contre une API FastAPI réelle.
// Prérequis : `uvicorn app.main:app --port 8000` dans backend/ ; lancer ensuite `npm test`.
import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Plotly a besoin d'un vrai navigateur : on le remplace par un composant qui vérifie les données.
vi.mock('../src/Chart.jsx', () => ({
  default: ({ data }) => {
    JSON.stringify(data)                       // lève si structure circulaire
    return <div data-testid="chart" />
  },
  COLORS: ['#4f46e5', '#0891b2', '#16a34a', '#d97706', '#db2777', '#7c3aed'], POS: '#dc2626', NEG: '#2563eb',
}))

import App from '../src/App.jsx'

const T = { timeout: 120000 }

describe('parcours complet', () => {
  it('données -> préparation -> modèles -> évaluation -> XAI -> API', async () => {
    const user = userEvent.setup()
    const errors = []
    vi.spyOn(console, 'error').mockImplementation((...a) => errors.push(a.join(' ')))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Télécharger et charger/ }))
    await screen.findByText('Aperçu et variable cible', {}, T)
    await user.click(screen.getByRole('button', { name: /Matrice de corrélation/ }))
    await waitFor(() => expect(screen.getAllByTestId('chart').length).toBeGreaterThan(4), T)

    await user.click(screen.getByRole('button', { name: /Passer à la préparation/ }))
    await user.click(await screen.findByRole('button', { name: /Préparer les données/ }))
    await screen.findByText('Résultat de la préparation', {}, T)

    await user.click(screen.getByRole('button', { name: /Passer à la modélisation/ }))
    await user.click(await screen.findByRole('button', { name: /Entraîner 5 modèle/ }))
    await screen.findByText('Modèles entraînés', {}, T)

    await user.click(screen.getByRole('button', { name: /Évaluer et comparer/ }))
    await screen.findByText('Tableau comparatif des modèles', {}, T)
    await screen.findByText(/Matrice de confusion/, {}, T)

    await user.click(screen.getByRole('button', { name: /Expliquer les décisions/ }))
    await screen.findByText('SHAP summary (beeswarm)', {}, T)
    await screen.findByText(/Importance par permutation/, {}, T)
    await waitFor(() => expect(document.body.textContent).toMatch(/baisse|Permutation/), T)

    await user.click(screen.getByRole('button', { name: /Explication d'un individu/ }))
    await user.click(await screen.findByRole('button', { name: /Expliquer cette prédiction/ }))
    await screen.findByText('Comparaison SHAP / LIME', {}, T)
    expect(document.querySelector('.sentence').textContent).toMatch(/^Le risque/)

    await user.click(screen.getByRole('button', { name: /Individu fictif/ }))
    await user.click(screen.getByRole('button', { name: /Expliquer cette prédiction/ }))
    await waitFor(() => expect(document.querySelector('.sentence')).toBeTruthy(), T)

    await user.click(screen.getByRole('button', { name: /Variables sensibles/ }))
    await screen.findByText('SeniorCitizen', {}, T)

    await user.click(screen.getByRole('button', { name: '6 · API' }))
    await screen.findByText(/Exemple d'appel/, {}, T)
    expect(errors.filter((e) => !/act\(/.test(e))).toEqual([])
  }, 300000)
})
