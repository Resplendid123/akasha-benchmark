import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from './api'
import type { Metrics, RootCause } from './types'

/** 缺失指标显示为 `—`，避免与数值 0 混淆。 */
export function metric(values: Metrics, name: string, digits = 4): string {
  const value = values[name]
  return value === undefined ? '—' : value.toFixed(digits)
}

export function percent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`
}

export function num(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits)
}

export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[],
): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError(null)
    Promise.resolve().then(load)
      .then((result) => {
        if (alive) setData(result)
      })
      .catch((exc: unknown) => {
        if (!alive) return
        setError(exc instanceof ApiError ? exc.message : String(exc))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  return { data, error, loading, reload: useCallback(() => setNonce((n) => n + 1), []) }
}

export function useAction<T>(): {
  run: (task: () => Promise<T>) => void
  busy: boolean
  error: string | null
  result: T | null
  reset: () => void
} {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<T | null>(null)
  const alive = useRef(true)
  useEffect(() => {
    // StrictMode 会重新执行 effect，每次挂载都要恢复异步更新标记。
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const run = useCallback((task: () => Promise<T>) => {
    setBusy(true)
    setError(null)
    setResult(null)
    Promise.resolve().then(task)
      .then((value) => alive.current && setResult(value))
      .catch((exc: unknown) =>
        alive.current && setError(exc instanceof ApiError ? exc.message : String(exc)),
      )
      .finally(() => alive.current && setBusy(false))
  }, [])

  return {
    run,
    busy,
    error,
    result,
    reset: useCallback(() => {
      setError(null)
      setResult(null)
    }, []),
  }
}

export function Loading({ what }: { what: string }) {
  return <p className="muted">正在加载{what}…</p>
}

export function Failed({ error }: { error: string }) {
  return (
    <div className="note bad">
      <strong>取数失败。</strong> {error}
    </div>
  )
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="muted">{children}</p>
}

/** answerMode 的显示。knowledge 之外的都要显眼 —— 它们的检索得分按定义为 0。 */
export function ModeTag({ mode }: { mode: string | null }) {
  const label = mode ?? 'missing'
  const kind = label === 'knowledge' ? 'ok' : 'warn'
  return <span className={`tag ${kind}`}>{label}</span>
}

export function Pass({ ok, yes = 'PASS', no = 'FAIL' }: { ok: boolean; yes?: string; no?: string }) {
  return <span className={`tag ${ok ? 'ok' : 'bad'}`}>{ok ? yes : no}</span>
}

export function Bar({ value, kind }: { value: number; kind?: 'ok' | 'bad' }) {
  return (
    <div className={`bar${kind ? ` ${kind}` : ''}`}>
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  )
}

/** 根因的中文名与色档。generation_fallback 是警告而非错误 —— 它不是检索问题。 */
export const ROOT_CAUSE_LABELS: Record<RootCause, { text: string; kind: string }> = {
  generation_fallback: { text: '生成端拒答', kind: 'warn' },
  compiled_away: { text: '编译丢词', kind: 'bad' },
  citation_dropped: { text: '引用被截断', kind: 'warn' },
  retrieval_miss: { text: '检索未命中', kind: 'bad' },
  graph_edge_missing: { text: '多跳缺跳', kind: 'bad' },
  gold_annotation_suspect: { text: '疑似标注问题', kind: '' },
  unknown: { text: '未能定位', kind: '' },
}

export function CauseTag({ cause }: { cause: string | null }) {
  if (!cause) return <span className="muted small">未归因</span>
  const entry = ROOT_CAUSE_LABELS[cause as RootCause]
  return <span className={`tag ${entry?.kind ?? ''}`}>{entry?.text ?? cause}</span>
}

/** 分页控件。列表体积不小，所以每个列表都分页。 */
export function Pager({
  total,
  offset,
  limit,
  onChange,
}: {
  total: number
  offset: number
  limit: number
  onChange: (offset: number) => void
}) {
  if (total <= limit) return null
  const page = Math.floor(offset / limit) + 1
  const pages = Math.ceil(total / limit)
  return (
    <div className="row tight small muted" style={{ marginTop: 10 }}>
      <button
        className="action small"
        disabled={offset === 0}
        onClick={() => onChange(Math.max(0, offset - limit))}
      >
        ← 上一页
      </button>
      <span>
        {page} / {pages}（共 {total} 条）
      </span>
      <button
        className="action small"
        disabled={offset + limit >= total}
        onClick={() => onChange(offset + limit)}
      >
        下一页 →
      </button>
    </div>
  )
}

/** 一个可折叠的段。链路视图里六跳内容全展开会太长。 */
export function Collapsible({
  title,
  children,
  open = false,
}: {
  title: React.ReactNode
  children: React.ReactNode
  open?: boolean
}) {
  const [shown, setShown] = useState(open)
  return (
    <div style={{ marginBottom: 10 }}>
      <button
        className="action small"
        onClick={() => setShown((s) => !s)}
        style={{ marginBottom: shown ? 8 : 0 }}
      >
        {shown ? '▾' : '▸'} {title}
      </button>
      {shown && children}
    </div>
  )
}

/** 数据集选择器。多选，因为指标勾选范围由所选组合决定。 */
export function DatasetPicker({
  all,
  selected,
  onChange,
}: {
  all: string[]
  selected: string[]
  onChange: (next: string[]) => void
}) {
  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name])
  return (
    <div className="row tight">
      {all.map((name) => (
        <label key={name} className="check">
          <input
            type="checkbox"
            checked={selected.includes(name)}
            onChange={() => toggle(name)}
          />
          {name}
        </label>
      ))}
    </div>
  )
}
