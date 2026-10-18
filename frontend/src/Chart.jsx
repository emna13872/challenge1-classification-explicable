import Plotly from 'plotly.js-cartesian-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

const grid = { gridcolor: '#e5e7eb', zerolinecolor: '#cbd5e1', linecolor: '#cbd5e1' }

export default function Chart({ data, layout = {}, height = 320 }) {
  const full = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#1f2937' },
    margin: { l: 55, r: 20, t: 30, b: 50 },
    legend: { orientation: 'h', y: -0.25 },
    height,
    ...layout,
    xaxis: { ...grid, ...(layout.xaxis || {}) },
    yaxis: { ...grid, ...(layout.yaxis || {}) },
  }
  return (
    <Plot data={data} layout={full} config={{ displaylogo: false, responsive: true }}
          useResizeHandler style={{ width: '100%' }} />
  )
}

export const COLORS = ['#4f46e5', '#0891b2', '#16a34a', '#d97706', '#db2777', '#7c3aed']
export const POS = '#dc2626'
export const NEG = '#2563eb'
