import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from './api'
import type { Metrics, RootCause, RunStatus, TaskStatus } from './types'

/** 缺失指标显示为 `—`，与数值 0 区分开。 */
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

/** 毫秒转成人读的时长，跨度从毫秒到小时。 */
export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || ms < 0) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m${rest.toString().padStart(2, '0')}s`
  return `${Math.floor(minutes / 60)}h${(minutes % 60).toString().padStart(2, '0')}m`
}

/** 两个时间戳之间的耗时。任务还在跑（没有 finished_at）时按当下算。 */
function elapsed(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt) return '—'
  const start = Date.parse(startedAt)
  if (Number.isNaN(start)) return '—'
  const end = finishedAt ? Date.parse(finishedAt) : Date.now()
  return Number.isNaN(end) ? '—' : duration(end - start)
}

/** 平均延迟 + 总耗时，如 ``2.6s/条 · 共 4.0s``。延迟缺失时只显示总耗时。 */
export function Timing({
  startedAt,
  finishedAt,
  latencyMs,
  perLabel = '每条',
}: {
  startedAt: string | null
  finishedAt: string | null
  latencyMs?: number | null
  perLabel?: string
}) {
  const total = elapsed(startedAt, finishedAt)
  const hasLatency = latencyMs !== null && latencyMs !== undefined
  return (
    <span className="small mono muted">
      {hasLatency && (
        <span title={`平均${perLabel} ${Math.round(latencyMs)}ms`}>
          {duration(latencyMs)}/{perLabel}
          {' · '}
        </span>
      )}
      {hasLatency ? `共 ${total}` : total}
    </span>
  )
}

/** 后端的 UTC 时间戳按 Asia/Shanghai 显示。 */
export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(date)
}

export function useAsync<T>(
  load: () => Promise<T>,
  deps: unknown[],
): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)
  const hasData = useRef(false)
  const inFlight = useRef(false)
  const queued = useRef(false)

  // 参数变了就回到加载态，手上的数据不再对应当前请求。
  useEffect(() => {
    hasData.current = false
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    let alive = true
    // 已有数据时刷新不翻转 loading，否则轮询会把整页反复换成加载态。
    if (!hasData.current) {
      setLoading(true)
      setError(null)
    }
    inFlight.current = true
    Promise.resolve()
      .then(load)
      .then((result) => {
        if (!alive) return
        hasData.current = true
        setData(result)
        setError(null)
      })
      .catch((exc: unknown) => {
        if (alive) setError(exc instanceof ApiError ? exc.message : String(exc))
      })
      .finally(() => {
        // 被后一次请求接替时交给接替者收尾。
        if (!alive) return
        inFlight.current = false
        setLoading(false)
        if (queued.current) {
          queued.current = false
          setNonce((n) => n + 1)
        }
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  // 上一次没回来就只记一次待刷新，既不堆积请求也不丢掉刷新。
  const reload = useCallback(() => {
    if (inFlight.current) {
      queued.current = true
      return
    }
    setNonce((n) => n + 1)
  }, [])

  return { data, error, loading, reload }
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
    Promise.resolve()
      .then(task)
      .then((value) => alive.current && setResult(value))
      .catch(
        (exc: unknown) =>
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

/** 有任务在跑时定时刷新。 */
export function usePoll(active: boolean, reload: () => void, ms = 3000) {
  useEffect(() => {
    if (!active) return
    const timer = setInterval(reload, ms)
    return () => clearInterval(timer)
  }, [active, reload, ms])
}

export function Loading({ what }: { what: string }) {
  return <p className="muted">正在加载{what}…</p>
}

export function Failed({ error }: { error: string }) {
  return (
    <div className="note bad">
      <strong>请求失败。</strong> {error}
    </div>
  )
}

export const STATUS_TEXT: Record<TaskStatus, string> = {
  queued: '排队中',
  running: '进行中',
  paused: '已暂停',
  succeeded: '已完成',
  failed: '失败',
}

const STATUS_CLASS: Record<TaskStatus, string> = {
  queued: 'tag',
  running: 'tag accent',
  paused: 'tag warn',
  succeeded: 'tag ok',
  failed: 'tag bad',
}

export function StatusTag({ status }: { status: TaskStatus | RunStatus }) {
  const key = status as TaskStatus
  return <span className={STATUS_CLASS[key] ?? 'tag'}>{STATUS_TEXT[key] ?? status}</span>
}

/** answerMode 的显示。非 knowledge 用警告色，它们的检索得分按定义为 0。 */
export function ModeTag({ mode }: { mode: string | null }) {
  const label = mode ?? 'missing'
  return <span className={`tag ${label === 'knowledge' ? 'ok' : 'warn'}`}>{label}</span>
}

export function Pass({ ok, yes = '通过', no = '未通过' }: { ok: boolean; yes?: string; no?: string }) {
  return <span className={`tag ${ok ? 'ok' : 'bad'}`}>{ok ? yes : no}</span>
}

export function Bar({ value, kind }: { value: number; kind?: 'ok' | 'bad' }) {
  return (
    <div className={`bar${kind ? ` ${kind}` : ''}`}>
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  )
}

/** 根因的中文名与色档。generation_fallback 用警告色，它不是检索问题。 */
const ROOT_CAUSE_LABELS: Record<RootCause, { text: string; kind: string }> = {
  // 唯一一个「没问题」的分类，用 ok 色。
  not_a_failure: { text: '答案正确', kind: 'ok' },
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

/** 分页控件。总数不超过一页时不渲染。 */
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
        {Math.floor(offset / limit) + 1} / {Math.ceil(total / limit)}（共 {total} 条）
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

/** 可折叠段。链路视图里内容全展开会太长。 */
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

/** 数据集多选。指标勾选范围由所选组合决定，所以是多选。 */
export function DatasetPicker({
  all,
  selected,
  onChange,
}: {
  all: string[]
  selected: string[]
  onChange: (next: string[]) => void
}) {
  return (
    <div className="row tight">
      {all.map((name) => (
        <label key={name} className="check">
          <input
            type="checkbox"
            checked={selected.includes(name)}
            onChange={() =>
              onChange(
                selected.includes(name)
                  ? selected.filter((n) => n !== name)
                  : [...selected, name],
              )
            }
          />
          {name}
        </label>
      ))}
    </div>
  )
}

export function Field({
  label,
  children,
  hint,
  wide,
}: {
  label: string
  children: React.ReactNode
  hint?: string
  /** 在 form-grid 里独占一行。 */
  wide?: boolean
}) {
  return (
    <label className={`field${wide ? ' wide' : ''}`}>
      <span>
        {label}
        {hint && <span className="muted"> · {hint}</span>}
      </span>
      {children}
    </label>
  )
}

/** 眼睛与划掉的眼睛。只这两个图标，不值得为它引一个图标库。 */
function EyeIcon({ off }: { off?: boolean }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {off ? (
        <>
          <path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c7 0 10 8 10 8a18.5 18.5 0 0 1-2.16 3.19m-2.72 2.42A9.7 9.7 0 0 1 12 20c-7 0-10-8-10-8a18.4 18.4 0 0 1 5.06-5.94" />
          <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
          <path d="M2 2l20 20" />
        </>
      ) : (
        <>
          <path d="M2 12s3-8 10-8 10 8 10 8-3 8-10 8-10-8-10-8z" />
          <circle cx="12" cy="12" r="3" />
        </>
      )}
    </svg>
  )
}

/** 密码与 api_key 统一走这里：默认遮住，自带显隐按钮。 */
export function SecretField({
  label,
  value,
  onChange,
  hint,
  placeholder,
  wide,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  hint?: string
  placeholder?: string
  wide?: boolean
}) {
  const [shown, setShown] = useState(false)
  return (
    <Field label={label} hint={hint} wide={wide}>
      <div className="secret-input">
        <input
          type={shown ? 'text' : 'password'}
          value={value}
          placeholder={placeholder}
          autoComplete="off"
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="secret-toggle"
          title={shown ? '隐藏' : '显示'}
          aria-label={shown ? '隐藏' : '显示'}
          onClick={() => setShown(!shown)}
        >
          <EyeIcon off={shown} />
        </button>
      </div>
    </Field>
  )
}

/** 清理按钮。二次确认写在这里一处，各层不各写一遍。 */
export function CleanupButton({
  what,
  detail,
  onConfirm,
  busy,
}: {
  what: string
  detail: string
  onConfirm: () => void
  busy?: boolean
}) {
  return (
    <button
      className="action small danger"
      disabled={busy}
      onClick={() => window.confirm(`清理${what}？\n\n${detail}`) && onConfirm()}
    >
      清理
    </button>
  )
}
