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

    // Comparaison par validation croisée : un « meilleur » doit apparaître, et le test ne doit pas avoir servi.
    await user.click(screen.getByRole('button', { name: /Comparer 5 modèle/ }))
    await screen.findByText('meilleur', {}, T)
    expect(document.body.textContent).toMatch(/Test utilisé : non/)

    // Calibration : la carte doit afficher les trois versions et rappeler que le test n'a pas servi.
    await user.click(screen.getByRole('button', { name: /Mesurer la fiabilité/ }))
    await screen.findByText(/sigmoïde \(Platt\)/, {}, T)
    await waitFor(() => expect(document.body.textContent).toMatch(/Brier de référence/), T)

    // Seuil de décision par le coût : le tableau compare le seuil de 50 % au meilleur seuil.
    await user.click(screen.getByRole('button', { name: /Chercher le meilleur seuil/ }))
    await screen.findByText(/Seuil 50 % \(par habitude\)/, {}, T)
    await waitFor(() => expect(document.body.textContent).toMatch(/Seuil retenu/), T)

    // Évaluation finale : le test s'ouvre une fois ; un second clic ne le relit pas.
    await user.click(screen.getByRole('button', { name: /Ouvrir le lot de test/ }))
    await screen.findByText(/choisi sans le test/, {}, T)
    await waitFor(() => expect(document.body.textContent).toMatch(/ouvert 1 fois/), T)
    await user.click(screen.getByRole('button', { name: /Revoir le résultat final/ }))
    await waitFor(() => expect(document.body.textContent).toMatch(/le test n'a pas été relu/), T)

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

    // Décision, motifs et recours : sur Telco aucune variable modifiable n'est déclarée, l'application doit le dire.
    await user.click(screen.getByRole('button', { name: /Motifs de la décision et conseil/ }))
    await screen.findByText('Décision et motifs', {}, T)
    await screen.findByText('Que changer pour être accepté ?', {}, T)
    await waitFor(() => expect(document.body.textContent).toMatch(/Variables sensibles jamais citées|Aucune variable modifiable|déjà accepté/), T)
    expect(document.querySelector('.sentence').textContent).toMatch(/^Le risque/)

    await user.click(screen.getByRole('button', { name: /Individu fictif/ }))
    await user.click(screen.getByRole('button', { name: /Expliquer cette prédiction/ }))
    await waitFor(() => expect(document.querySelector('.sentence')).toBeTruthy(), T)

    await user.click(screen.getByRole('button', { name: /Variables sensibles/ }))
    await screen.findByText('SeniorCitizen', {}, T)

    // Audit du recours : sur Telco rien n'est déclaré modifiable, l'audit doit le dire au lieu d'inventer des chiffres.
    await user.click(screen.getByRole('button', { name: /Audit du recours/ }))
    await user.click(await screen.findByRole('button', { name: /Lancer l'audit/ }))
    await waitFor(() => expect(document.body.textContent).toMatch(/Aucune variable modifiable n'est déclarée|Aucun client refusé/), T)

    await user.click(screen.getByRole('button', { name: '6 · API' }))
    await screen.findByText(/Exemple d'appel/, {}, T)
    expect(errors.filter((e) => !/act\(/.test(e))).toEqual([])
  }, 300000)
})
