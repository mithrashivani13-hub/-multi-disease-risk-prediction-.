import { useState, useRef, useEffect, type CSSProperties, type ReactNode } from 'react'

// ─── API config ───────────────────────────────────────────────────────────────
// Flask backend from src/app.py. Change this if you deploy the API elsewhere.
const API_BASE = 'http://localhost:5000'

// ─── Palette ──────────────────────────────────────────────────────────────────
const C = {
  navy:   '#1B3A4B',
  green:  '#0EA679',
  red:    '#D9483C',
  bg:     '#F5F7F6',
  border: '#C5D2D9',
  muted:  '#7A90A0',
  card:   '#FFFFFF',
  text:   '#1A2B35',
  sub:    '#4B6474',
} as const

// ─── Types ────────────────────────────────────────────────────────────────────
type Tab = 'diabetes' | 'heart'

interface Prediction {
  disease: Tab
  probability: number
  isAtRisk: boolean
  model: string
}

// Raw shape returned by /predict/diabetes and /predict/heart
interface ApiResponse {
  disease: string
  prediction: 'At Risk' | 'Low Risk'
  risk_probability: number
  model_used: string
  error?: string
}

type NumField  = { kind: 'number';  name: string; label: string; placeholder: string; step?: string }
type SelField  = { kind: 'select';  name: string; label: string; options: { value: string; label: string }[] }
type FieldDef  = NumField | SelField

// ─── Field Definitions ────────────────────────────────────────────────────────
// `name` is the camelCase key used by the form. `apiKey` is the key the Flask
// API actually expects (see src/predict.py / models/*_meta.json feature_order).
const DIABETES_FIELDS: (FieldDef & { apiKey: string })[] = [
  { kind: 'number', name: 'pregnancies',     apiKey: 'Pregnancies',               label: 'Pregnancies',                 placeholder: '2'     },
  { kind: 'number', name: 'glucose',         apiKey: 'Glucose',                   label: 'Glucose (mg/dL)',             placeholder: '120'   },
  { kind: 'number', name: 'bloodPressure',   apiKey: 'BloodPressure',             label: 'Blood Pressure (mmHg)',       placeholder: '70'    },
  { kind: 'number', name: 'skinThickness',   apiKey: 'SkinThickness',             label: 'Skin Thickness (mm)',         placeholder: '23'    },
  { kind: 'number', name: 'insulin',         apiKey: 'Insulin',                   label: 'Insulin (μU/mL)',             placeholder: '80'    },
  { kind: 'number', name: 'bmi',             apiKey: 'BMI',                       label: 'BMI',                         placeholder: '28.5', step: '0.1' },
  { kind: 'number', name: 'diabetesPedigree',apiKey: 'DiabetesPedigreeFunction',  label: 'Diabetes Pedigree Function',  placeholder: '0.472',step: '0.001' },
  { kind: 'number', name: 'age',             apiKey: 'Age',                       label: 'Age (years)',                 placeholder: '35'    },
]

const HEART_FIELDS: (FieldDef & { apiKey: string })[] = [
  { kind: 'number', name: 'age',      apiKey: 'age',      label: 'Age (years)',                  placeholder: '54'  },
  { kind: 'select', name: 'sex',      apiKey: 'sex',      label: 'Sex',                          options: [{ value: '1', label: 'Male' }, { value: '0', label: 'Female' }] },
  { kind: 'select', name: 'cp',       apiKey: 'cp',       label: 'Chest Pain Type',              options: [
    { value: '0', label: 'Typical Angina'   },
    { value: '1', label: 'Atypical Angina'  },
    { value: '2', label: 'Non-Anginal Pain' },
    { value: '3', label: 'Asymptomatic'     },
  ]},
  { kind: 'number', name: 'trestbps', apiKey: 'trestbps', label: 'Resting BP (mmHg)',            placeholder: '130' },
  { kind: 'number', name: 'chol',     apiKey: 'chol',     label: 'Cholesterol (mg/dL)',          placeholder: '240' },
  { kind: 'select', name: 'fbs',      apiKey: 'fbs',      label: 'Fasting Blood Sugar > 120 mg/dL', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }] },
  { kind: 'select', name: 'restecg',  apiKey: 'restecg',  label: 'Resting ECG',                  options: [
    { value: '0', label: 'Normal'                    },
    { value: '1', label: 'ST-T Wave Abnormality'     },
    { value: '2', label: 'Left Ventricular Hypertrophy' },
  ]},
  { kind: 'number', name: 'thalach',  apiKey: 'thalach',  label: 'Max Heart Rate (bpm)',         placeholder: '152' },
  { kind: 'select', name: 'exang',    apiKey: 'exang',    label: 'Exercise-Induced Angina',      options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }] },
  { kind: 'number', name: 'oldpeak',  apiKey: 'oldpeak',  label: 'Oldpeak (ST Depression)',      placeholder: '1.0', step: '0.1' },
  { kind: 'select', name: 'slope',    apiKey: 'slope',    label: 'Slope of Peak ST Segment',     options: [
    { value: '0', label: 'Upsloping'   },
    { value: '1', label: 'Flat'        },
    { value: '2', label: 'Downsloping' },
  ]},
  { kind: 'number', name: 'ca',       apiKey: 'ca',       label: 'Major Vessels (0–3)',          placeholder: '0'   },
  { kind: 'select', name: 'thal',     apiKey: 'thal',     label: 'Thal',                         options: [
    { value: '1', label: 'Normal'           },
    { value: '2', label: 'Fixed Defect'     },
    { value: '3', label: 'Reversible Defect'},
  ]},
]

const FIELD_MAP: Record<Tab, (FieldDef & { apiKey: string })[]> = {
  diabetes: DIABETES_FIELDS,
  heart: HEART_FIELDS,
}

// ─── API call ─────────────────────────────────────────────────────────────────
async function runPrediction(tab: Tab, data: Record<string, string>): Promise<Prediction> {
  const fields = FIELD_MAP[tab]

  // Build the payload with the exact keys/types the Flask API expects.
  const payload: Record<string, number> = {}
  for (const f of fields) {
    const raw = data[f.name] ?? (f.kind === 'select' ? f.options[0].value : '0')
    payload[f.apiKey] = f.kind === 'number' ? parseFloat(raw) || 0 : parseInt(raw, 10) || 0
  }

  const endpoint = tab === 'diabetes' ? '/predict/diabetes' : '/predict/heart'
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  const json: ApiResponse = await res.json()
  if (!res.ok) {
    throw new Error(json.error || `Request failed (${res.status})`)
  }

  return {
    disease: tab,
    probability: json.risk_probability,
    isAtRisk: json.prediction === 'At Risk',
    model: json.model_used,
  }
}

// ─── ECG Strip ────────────────────────────────────────────────────────────────
function ECGStrip() {
  const h = 36, mid = 18, CW = 90, N = 38
  const W = CW * N  // 3420px per half

  let d = `M0,${mid}`
  for (let i = 0; i < N * 2; i++) {
    const o = i * CW
    d += ` L${o+8},${mid}`
    d += ` L${o+10},${mid-2} L${o+12},${mid-4} L${o+14},${mid-2} L${o+15},${mid}`
    d += ` L${o+16},${mid+1} L${o+17},${mid-13} L${o+19},${mid+12} L${o+21},${mid}`
    d += ` L${o+27},${mid} L${o+29},${mid-5} L${o+34},${mid-5} L${o+38},${mid}`
    d += ` L${o+90},${mid}`
  }

  const svgAnim: CSSProperties = {
    position: 'absolute', top: 0, left: 0,
    animation: 'ecgScroll 18s linear infinite',
  }

  return (
    <div style={{ position: 'relative', height: h, background: C.navy, overflow: 'hidden', flexShrink: 0 }}>
      <svg
        style={{ ...svgAnim, display: 'block', width: W * 2, height: h }}
        viewBox={`0 0 ${W * 2} ${h}`}
        preserveAspectRatio="xMinYMid meet"
      >
        <path d={d} fill="none" stroke="rgba(255,255,255,0.11)" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      </svg>

      <div style={{
        position: 'absolute', inset: 0,
        maskImage: 'linear-gradient(90deg, transparent 0%, white 25%, white 75%, transparent 100%)',
        WebkitMaskImage: 'linear-gradient(90deg, transparent 0%, white 25%, white 75%, transparent 100%)',
        maskSize: '28% 100%',
        WebkitMaskSize: '28% 100%',
        maskRepeat: 'no-repeat',
        WebkitMaskRepeat: 'no-repeat',
        animation: 'ecgPulse 5s linear infinite',
      }}>
        <svg
          style={{ ...svgAnim, display: 'block', width: W * 2, height: h }}
          viewBox={`0 0 ${W * 2} ${h}`}
          preserveAspectRatio="xMinYMid meet"
        >
          <defs>
            <filter id="ecgGlow" x="-10%" y="-60%" width="120%" height="220%">
              <feGaussianBlur stdDeviation="1.8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path d={d} fill="none" stroke={C.green} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" filter="url(#ecgGlow)" />
        </svg>
      </div>
    </div>
  )
}

// ─── Reusable Components ──────────────────────────────────────────────────────

function FormField({ field, value, onChange }: {
  field: FieldDef
  value: string
  onChange: (name: string, val: string) => void
}) {
  const labelStyle: CSSProperties = {
    display: 'block',
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: C.muted,
    marginBottom: 4,
    fontFamily: "'Inter', sans-serif",
  }
  const baseInput: CSSProperties = {
    width: '100%',
    height: 34,
    padding: '0 10px',
    fontSize: 13,
    fontFamily: "'IBM Plex Mono', monospace",
    color: C.text,
    background: C.bg,
    border: `1px solid ${C.border}`,
    borderRadius: 2,
    transition: 'border-color 0.15s, box-shadow 0.15s',
  }

  return (
    <div>
      <label style={labelStyle}>{field.label}</label>
      {field.kind === 'number' ? (
        <input
          type="number"
          className="vt-input"
          style={baseInput}
          placeholder={field.placeholder}
          step={field.step ?? '1'}
          value={value}
          onChange={e => onChange(field.name, e.target.value)}
        />
      ) : (
        <select
          className="vt-select"
          style={{ ...baseInput, appearance: 'none', backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%237A90A0' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 10px center', paddingRight: 28, cursor: 'pointer' }}
          value={value}
          onChange={e => onChange(field.name, e.target.value)}
        >
          {field.options.map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      )}
    </div>
  )
}

function TabSwitcher({ active, onChange }: { active: Tab; onChange: (t: Tab) => void }) {
  return (
    <div style={{
      display: 'inline-flex',
      background: C.bg,
      border: `1px solid ${C.border}`,
      borderRadius: 3,
      padding: 2,
      marginBottom: 20,
    }}>
      {(['diabetes', 'heart'] as Tab[]).map(tab => (
        <button
          key={tab}
          onClick={() => onChange(tab)}
          style={{
            padding: '5px 16px',
            fontSize: 12,
            fontWeight: 500,
            fontFamily: "'Inter', sans-serif",
            letterSpacing: '0.02em',
            borderRadius: 2,
            border: 'none',
            cursor: 'pointer',
            transition: 'background 0.15s, color 0.15s',
            background: active === tab ? C.navy : 'transparent',
            color: active === tab ? '#fff' : C.muted,
          }}
        >
          {tab === 'diabetes' ? 'Diabetes' : 'Heart Disease'}
        </button>
      ))}
    </div>
  )
}

function PrimaryButton({ children, onClick, loading }: {
  children: ReactNode
  onClick: () => void
  loading?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        width: '100%',
        height: 40,
        background: loading ? '#3A6070' : C.navy,
        color: '#fff',
        border: 'none',
        borderRadius: 2,
        fontSize: 13,
        fontWeight: 500,
        fontFamily: "'Inter', sans-serif",
        letterSpacing: '0.02em',
        cursor: loading ? 'not-allowed' : 'pointer',
        marginTop: 20,
        transition: 'background 0.15s',
      }}
    >
      {loading ? (
        <>
          <span style={{ display: 'inline-block', width: 14, height: 14, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
          Processing…
        </>
      ) : children}
    </button>
  )
}

function VerdictBadge({ isAtRisk }: { isAtRisk: boolean }) {
  const color = isAtRisk ? C.red : C.green
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 7,
      padding: '5px 14px',
      border: `1.5px solid ${color}`,
      borderRadius: 20,
      background: `${color}14`,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0 }} />
      <span style={{
        fontSize: 11,
        fontWeight: 700,
        fontFamily: "'Inter', sans-serif",
        letterSpacing: '0.10em',
        textTransform: 'uppercase',
        color,
      }}>
        {isAtRisk ? 'At Risk' : 'Low Risk'}
      </span>
    </div>
  )
}

function CircularGauge({ pct, isAtRisk, animated }: { pct: number; isAtRisk: boolean; animated: boolean }) {
  const r = 52
  const cx = 64, cy = 64
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - pct / 100)
  const color = isAtRisk ? C.red : C.green
  const gaugeRef = useRef<SVGCircleElement>(null)

  useEffect(() => {
    if (!animated || !gaugeRef.current) return
    gaugeRef.current.style.transition = 'none'
    gaugeRef.current.style.strokeDashoffset = String(circ)
    const frame = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (gaugeRef.current) {
          gaugeRef.current.style.transition = 'stroke-dashoffset 1.1s cubic-bezier(0.4, 0, 0.2, 1)'
          gaugeRef.current.style.strokeDashoffset = String(offset)
        }
      })
    })
    return () => cancelAnimationFrame(frame)
  }, [pct, animated, offset, circ])

  return (
    <div style={{ position: 'relative', width: 128, height: 128 }}>
      <svg
        viewBox="0 0 128 128"
        style={{ transform: 'rotate(-90deg)', display: 'block', width: '100%', height: '100%' }}
      >
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#E2EAED" strokeWidth="7" />
        <circle
          ref={gaugeRef}
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={animated ? circ : offset}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: 1,
      }}>
        <span style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 22,
          fontWeight: 600,
          color: C.text,
          lineHeight: 1,
        }}>
          {(pct).toFixed(1)}%
        </span>
        <span style={{ fontSize: 9, fontFamily: "'Inter', sans-serif", color: C.muted, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          Risk Score
        </span>
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 14,
      minHeight: 320,
      color: C.muted,
    }}>
      <svg viewBox="0 0 128 128" style={{ width: 128, height: 128, opacity: 0.35 }}>
        <circle cx="64" cy="64" r="52" fill="none" stroke={C.border} strokeWidth="7" strokeDasharray="6 6" />
        <circle cx="64" cy="64" r="32" fill="none" stroke={C.border} strokeWidth="1" />
        <line x1="64" y1="34" x2="64" y2="94" stroke={C.border} strokeWidth="1" />
        <line x1="34" y1="64" x2="94" y2="64" stroke={C.border} strokeWidth="1" />
      </svg>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 13, fontWeight: 500, fontFamily: "'Inter', sans-serif", color: C.sub, marginBottom: 4 }}>
          No assessment run
        </p>
        <p style={{ fontSize: 12, fontFamily: "'Inter', sans-serif", color: C.muted }}>
          Complete the patient intake form and run an assessment to see the prediction.
        </p>
      </div>
    </div>
  )
}

function ErrorState({ message }: { message: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 12,
      minHeight: 320,
      textAlign: 'center',
      padding: '0 12px',
    }}>
      <span style={{
        width: 40, height: 40, borderRadius: '50%',
        background: `${C.red}14`, border: `1.5px solid ${C.red}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: C.red, fontSize: 18, fontWeight: 700,
      }}>!</span>
      <p style={{ fontSize: 13, fontWeight: 500, fontFamily: "'Inter', sans-serif", color: C.sub }}>
        Couldn't reach the prediction API
      </p>
      <p style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: C.muted, maxWidth: 320 }}>
        {message}
      </p>
      <p style={{ fontSize: 11, fontFamily: "'Inter', sans-serif", color: C.muted, maxWidth: 320 }}>
        Make sure the Flask backend is running: <code>python3 src/app.py</code>
      </p>
    </div>
  )
}

function ResultCard({ result }: { result: Prediction }) {
  const pct = result.probability * 100
  const color = result.isAtRisk ? C.red : C.green
  const diseaseName = result.disease === 'diabetes' ? 'Diabetes' : 'Heart Disease'

  return (
    <div style={{ animation: 'fadeSlideIn 0.35s ease' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, paddingTop: 12, paddingBottom: 24 }}>
        <CircularGauge pct={pct} isAtRisk={result.isAtRisk} animated />
        <VerdictBadge isAtRisk={result.isAtRisk} />
      </div>

      <div style={{ borderTop: `1px solid ${C.border}` }}>
        {[
          { label: 'Disease',     value: diseaseName },
          { label: 'Model',       value: result.model },
          { label: 'Probability', value: `${(result.probability * 100).toFixed(2)}%`, mono: true },
          { label: 'Threshold',   value: '≥ 50.00%', mono: true },
          { label: 'Verdict',     value: result.isAtRisk ? 'Positive' : 'Negative', color },
        ].map(row => (
          <div key={row.label} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            padding: '9px 0',
            borderBottom: `1px solid ${C.border}`,
          }}>
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.muted, fontFamily: "'Inter', sans-serif" }}>
              {row.label}
            </span>
            <span style={{
              fontSize: 13,
              fontFamily: row.mono ? "'IBM Plex Mono', monospace" : "'Inter', sans-serif",
              fontWeight: 500,
              color: row.color ?? C.text,
            }}>
              {row.value}
            </span>
          </div>
        ))}
      </div>

      <div style={{
        marginTop: 16,
        padding: '10px 12px',
        background: result.isAtRisk ? `${C.red}08` : `${C.green}08`,
        border: `1px solid ${color}30`,
        borderRadius: 2,
      }}>
        <p style={{ fontSize: 11, color: C.sub, fontFamily: "'Inter', sans-serif", lineHeight: 1.55, margin: 0 }}>
          {result.isAtRisk
            ? 'Elevated risk detected. Clinical follow-up and further diagnostic evaluation are recommended. This result is not a clinical diagnosis.'
            : 'Risk indicators within normal range. Routine monitoring is advised. This result is not a clinical diagnosis.'}
        </p>
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('diabetes')
  const [diabetesData, setDiabetesData] = useState<Record<string, string>>({})
  const [heartData, setHeartData] = useState<Record<string, string>>({})
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const currentData = activeTab === 'diabetes' ? diabetesData : heartData
  const setCurrentData = activeTab === 'diabetes' ? setDiabetesData : setHeartData
  const currentFields = FIELD_MAP[activeTab]

  const handleFieldChange = (name: string, val: string) => {
    setCurrentData(prev => ({ ...prev, [name]: val }))
  }

  const handleSubmit = async () => {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const pred = await runPrediction(activeTab, currentData)
      setResult(pred)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const sectionLabel: CSSProperties = {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: '0.10em',
    textTransform: 'uppercase',
    color: C.muted,
    fontFamily: "'Inter', sans-serif",
  }

  const cardStyle: CSSProperties = {
    background: C.card,
    border: `1px solid ${C.border}`,
    borderRadius: 3,
    padding: '20px 24px 24px',
    display: 'flex',
    flexDirection: 'column',
  }

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: "'Inter', sans-serif", color: C.text }}>

      <ECGStrip />

      <header style={{
        background: C.card,
        borderBottom: `1px solid ${C.border}`,
        padding: '14px 32px',
        display: 'flex',
        alignItems: 'baseline',
        gap: 12,
      }}>
        <span style={{
          fontFamily: "'Libre Caslon Text', Georgia, serif",
          fontStyle: 'italic',
          fontSize: 22,
          fontWeight: 700,
          color: C.navy,
          letterSpacing: '-0.01em',
          lineHeight: 1,
        }}>
          Vitalign
        </span>
        <span style={{ fontSize: 11, color: C.muted, fontWeight: 400, letterSpacing: '0.04em' }}>
          Multi-Disease Risk Assessment Console
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ ...sectionLabel, fontSize: 9 }}>v2.4.1</span>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.green, display: 'inline-block' }} />
          <span style={{ ...sectionLabel, fontSize: 9, color: C.green }}>System online</span>
        </div>
      </header>

      <main style={{
        maxWidth: 1200,
        margin: '0 auto',
        padding: '28px 24px 48px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))',
        gap: 20,
        alignItems: 'start',
      }}>

        <div style={cardStyle}>
          <div style={{ marginBottom: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h2 style={{
              fontFamily: "'Libre Caslon Text', Georgia, serif",
              fontStyle: 'italic',
              fontSize: 18,
              fontWeight: 700,
              color: C.navy,
              margin: 0,
              letterSpacing: '-0.01em',
            }}>
              Patient Intake
            </h2>
            <span style={{ ...sectionLabel }}>
              {activeTab === 'diabetes' ? 'Diabetes Screening' : 'Cardiac Screening'}
            </span>
          </div>

          <div style={{ height: 1, background: C.border, margin: '12px 0 16px' }} />

          <TabSwitcher active={activeTab} onChange={tab => { setActiveTab(tab); setResult(null); setError(null) }} />

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gap: '12px 16px',
            overflowY: 'auto',
            maxHeight: '56vh',
            paddingRight: 4,
          }}>
            {currentFields.map(field => (
              <div
                key={field.name}
                style={{ gridColumn: field.kind === 'select' && field.options.length > 3 ? 'span 2' : undefined }}
              >
                <FormField
                  field={field}
                  value={currentData[field.name] ?? (field.kind === 'select' ? field.options[0].value : '')}
                  onChange={handleFieldChange}
                />
              </div>
            ))}
          </div>

          <PrimaryButton onClick={handleSubmit} loading={loading}>
            Run Risk Assessment
          </PrimaryButton>
        </div>

        <div style={cardStyle}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <h2 style={{
              fontFamily: "'Libre Caslon Text', Georgia, serif",
              fontStyle: 'italic',
              fontSize: 18,
              fontWeight: 700,
              color: C.navy,
              margin: 0,
              letterSpacing: '-0.01em',
            }}>
              Prediction Results
            </h2>
            {result && (
              <span style={{ ...sectionLabel, color: result.isAtRisk ? C.red : C.green }}>
                {result.disease === 'diabetes' ? 'Diabetes' : 'Heart Disease'}
              </span>
            )}
          </div>

          <div style={{ height: 1, background: C.border, margin: '12px 0 16px' }} />

          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20, minHeight: 320 }}>
              <div style={{ display: 'flex', gap: 5 }}>
                {[0, 1, 2, 3].map(i => (
                  <span key={i} style={{
                    display: 'block', width: 6, height: 6, borderRadius: '50%', background: C.navy,
                    animation: `loadPulse 1s ease-in-out ${i * 0.15}s infinite`,
                  }} />
                ))}
              </div>
              <p style={{ fontSize: 12, color: C.muted, fontFamily: "'IBM Plex Mono', monospace", letterSpacing: '0.04em' }}>
                Running model inference…
              </p>
            </div>
          ) : error ? (
            <ErrorState message={error} />
          ) : result ? (
            <ResultCard result={result} />
          ) : (
            <EmptyState />
          )}
        </div>

      </main>

      <footer style={{
        borderTop: `1px solid ${C.border}`,
        background: C.card,
        padding: '12px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 8,
      }}>
        <span style={{ fontSize: 11, color: C.muted, fontFamily: "'IBM Plex Mono', monospace" }}>
          Vitalign — Research & Educational Use Only
        </span>
        <span style={{ fontSize: 11, color: C.muted, fontFamily: "'Inter', sans-serif" }}>
          Models: Random Forest · K-Nearest Neighbors · PIMA · Cleveland Heart datasets
        </span>
      </footer>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes loadPulse {
          0%, 100% { opacity: 0.25; transform: scale(0.85); }
          50%       { opacity: 1;    transform: scale(1); }
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
