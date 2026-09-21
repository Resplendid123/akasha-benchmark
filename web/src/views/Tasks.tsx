import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Task, TaskDetail } from '../types'
import {
  Bar,
  Failed,
  Loading,
  Pager,
  STATUS_TEXT,
  StatusTag,
  formatDateTime,
  useAction,
  useAsync,
  usePoll,
} from '../ui'

/** 任务层：实时观测六层的任务，支持暂停、继续、清理。 */
export function Tasks() {
  const PAGE_SIZE = 10
  const [offset, setOffset] = useState(0)
  const tasks = useAsync(() => api.tasks({ limit: PAGE_SIZE, offset }), [offset])
  const stages = useAsync(() => api.stages(), [])
  const [open, setOpen] = useState<number | null>(null)
  const [showAudit, setShowAudit] = useState(false)
  const cleanup = useAction<{ deleted: number }>()

  const list = tasks.data?.tasks ?? []
  usePoll(
    list.some((t) => t.status === 'running' || t.status === 'queued'),
    tasks.reload,
  )

  const removableCount = tasks.data?.inactive_total ?? 0

  useEffect(() => {
    const total = tasks.data?.total
    if (total !== undefined && total > 0 && offset >= total) {
      setOffset(Math.floor((total - 1) / PAGE_SIZE) * PAGE_SIZE)
    }
  }, [offset, tasks.data?.total])

  return (
    <>
      <h2>任务</h2>

      <div className="spread" style={{ marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>任务列表</h3>
        <div className="row tight">
          <button className="action small" onClick={tasks.reload}>
            刷新
          </button>
          <button className="action small" onClick={() => setShowAudit(!showAudit)}>
            {showAudit ? '收起审计日志' : '审计日志'}
          </button>
          <button
            className="action small danger"
            disabled={removableCount === 0 || cleanup.busy}
            onClick={() => {
              if (!window.confirm('清理所有已结束的任务记录？审计日志会保留。')) return
              cleanup.run(async () => {
                const result = await api.cleanupTasks()
                if (offset === 0) tasks.reload()
                else setOffset(0)
                return result
              })
            }}
          >
            清理已结束（{removableCount}）
          </button>
        </div>
      </div>

      {tasks.loading && <Loading what="任务" />}
      {tasks.error && <Failed error={tasks.error} />}
      {tasks.data?.total === 0 && !tasks.loading && <p className="muted">还没有任务。</p>}

      {list.length > 0 && (
        <>
          <table className="records-table tasks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>阶段</th>
                <th>状态</th>
                <th>进度</th>
                <th>参数</th>
                <th>开始</th>
                <th>结束</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {list.map((task) => (
                <Row
                  key={task.id}
                  task={task}
                  label={stages.data?.find((s) => s.stage === task.stage)?.label ?? task.stage}
                  open={open === task.id}
                  onToggle={() => setOpen(open === task.id ? null : task.id)}
                  onChanged={tasks.reload}
                />
              ))}
            </tbody>
          </table>
          <Pager
            total={tasks.data?.total ?? 0}
            offset={offset}
            limit={PAGE_SIZE}
            onChange={(next) => {
              setOpen(null)
              setOffset(next)
            }}
          />
        </>
      )}

      {cleanup.error && <Failed error={cleanup.error} />}

      {open !== null && <Logs key={open} taskId={open} onClose={() => setOpen(null)} />}
      {showAudit && <Audit />}
    </>
  )
}

function Row({
  task,
  label,
  open,
  onToggle,
  onChanged,
}: {
  task: Task
  label: string
  open: boolean
  onToggle: () => void
  onChanged: () => void
}) {
  const action = useAction<unknown>()
  const active = task.status === 'running' || task.status === 'queued'
  const resumable = task.status === 'paused' || task.status === 'failed'
  const ratio =
    task.status === 'succeeded'
      ? 1
      : task.progress_total && task.progress_total > 0
        ? task.progress_done / task.progress_total
        : null

  const act = (fn: () => Promise<unknown>) =>
    action.run(async () => {
      const result = await fn()
      onChanged()
      return result
    })

  return (
    <tr className={open ? 'selected' : ''}>
      <td className="mono">{task.id}</td>
      <td>{label}</td>
      <td>
        <StatusTag status={task.status} />
      </td>
      <td style={{ minWidth: 130 }}>
        {ratio !== null ? (
          <>
            <Bar value={ratio} kind={task.status === 'failed' ? 'bad' : undefined} />
            <span className="small mono muted">
              {task.progress_done}/{task.progress_total}
            </span>
          </>
        ) : (
          <span className="small mono muted">{task.progress_done || '—'}</span>
        )}
        {task.progress_note && <span className="small muted"> {task.progress_note}</span>}
      </td>
      <td className="small mono muted truncate" title={JSON.stringify(task.params)}>
        {Object.entries(task.params)
          .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join('+') : value}`)
          .join(' ') || '—'}
      </td>
      <td className="small muted mono">{formatDateTime(task.started_at)}</td>
      <td className="small muted mono">{formatDateTime(task.finished_at)}</td>
      <td className="table-actions-cell">
        <div className="table-actions">
          <button className="action small" onClick={onToggle}>
            {open ? '收起' : '日志'}
          </button>
          {active && (
            <button
              className="action small"
              disabled={action.busy}
              title="停在下一个可续跑的边界；已提交给 Akasha 的编译不会因此停止"
              onClick={() => act(() => api.pauseTask(task.id))}
            >
              暂停
            </button>
          )}
          {resumable && (
            <button
              className="action small primary"
              disabled={action.busy}
              onClick={() => act(() => api.resumeTask(task.id))}
            >
              继续
            </button>
          )}
          {!active && (
            <button
              className="action small danger"
              disabled={action.busy}
              title="删任务记录，审计日志保留"
              onClick={() => act(() => api.deleteTask(task.id))}
            >
              清理
            </button>
          )}
        </div>
        {action.error && (
          <div className="small" style={{ color: 'var(--bad)' }}>
            {action.error}
          </div>
        )}
      </td>
    </tr>
  )
}

/** 增量日志，只拉 after_id 之后新增的行。 */
function Logs({ taskId, onClose }: { taskId: number; onClose: () => void }) {
  const [detail, setDetail] = useState<TaskDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    let lastId = 0
    let active = true

    const tick = () => {
      api
        .task(taskId, lastId)
        .then((next) => {
          if (!alive) return
          if (next.logs.length > 0) lastId = next.logs[next.logs.length - 1]!.id
          active = next.status === 'running' || next.status === 'queued'
          setDetail((previous) =>
            previous ? { ...next, logs: [...previous.logs, ...next.logs].slice(-500) } : next,
          )
        })
        .catch((exc: unknown) => {
          if (alive) setError(exc instanceof Error ? exc.message : String(exc))
        })
    }

    tick()
    const timer = setInterval(() => active && tick(), 2000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [taskId])

  if (error) return <Failed error={error} />
  if (!detail) return <Loading what="任务日志" />

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>
          任务 #{detail.id} · {detail.stage} <StatusTag status={detail.status} />
        </h3>
        <div className="row tight small muted">
          {detail.target_kind && (
            <span className="mono">
              {detail.target_kind} #{detail.target_id}
            </span>
          )}
          <button className="action small" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>

      {detail.error && <div className="note bad">{detail.error}</div>}

      <pre className="block tall" style={{ marginTop: 10 }}>
        {detail.logs.map((entry) => `[${entry.level}] ${entry.message}`).join('\n') ||
          '（还没有输出）'}
      </pre>
      <p className="small muted">
        {STATUS_TEXT[detail.status]} · 参数 {JSON.stringify(detail.params)}
      </p>
    </div>
  )
}

/** 审计日志。只追加，清理任务不删它。 */
function Audit() {
  const { data, error, loading } = useAsync(() => api.audit(), [])
  if (loading) return <Loading what="审计日志" />
  if (error) return <Failed error={error} />
  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>审计日志</h3>
      <pre className="block tall">
        {(data ?? [])
          .map((entry) => `${entry.at} [${entry.stage}/${entry.level}] ${entry.message}`)
          .join('\n') || '（还没有记录）'}
      </pre>
    </div>
  )
}
