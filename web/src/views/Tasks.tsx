import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Task, TaskDetail } from '../types'
import { Bar, Failed, Loading, useAction, useAsync } from '../ui'

const STATUS_CLASS: Record<Task['status'], string> = {
  queued: 'tag',
  running: 'tag accent',
  succeeded: 'tag ok',
  failed: 'tag bad',
  cancelled: 'tag warn',
}

const STATUS_TEXT: Record<Task['status'], string> = {
  queued: '排队中',
  running: '进行中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已停止',
}

export function Tasks() {
  const tasks = useAsync(() => api.tasks(), [])
  const stages = useAsync(() => api.stages(), [])
  const [openTask, setOpenTask] = useState<number | null>(null)
  const cleanup = useAction<{ deleted: number }>()

  // 有任务在跑就每 3 秒刷一次，进度逐行提交。
  useEffect(() => {
    const active = tasks.data?.some((t) => t.status === 'running' || t.status === 'queued')
    if (!active) return
    const timer = setInterval(tasks.reload, 3000)
    return () => clearInterval(timer)
  }, [tasks.data, tasks.reload])

  const finished = (tasks.data ?? []).filter(
    (t) => t.status !== 'running' && t.status !== 'queued',
  )

  return (
    <>
      <h2>任务</h2>

      {stages.data && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>阶段描述</h3>
          <table>
            <thead>
              <tr>
                <th>阶段</th>
                <th>代价</th>
                <th>需要 Akasha 启动</th>
              </tr>
            </thead>
            <tbody>
              {stages.data.map((stage) => (
                <tr key={stage.stage}>
                  <td>
                    {stage.label} <span className="small mono muted">{stage.stage}</span>
                  </td>
                  <td className="small muted">{stage.cost}</td>
                  <td>{stage.needs_akasha ? <span className="tag warn">是</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="spread" style={{ marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>任务列表</h3>
        <div className="row tight">
          <button className="action small" onClick={tasks.reload}>
            刷新
          </button>
          <button
            className="action small danger"
            disabled={finished.length === 0 || cleanup.busy}
            onClick={() =>
              cleanup.run(async () => {
                const result = await api.cleanupFinished()
                tasks.reload()
                return result
              })
            }
          >
            清理已结束（{finished.length}）
          </button>
        </div>
      </div>

      {cleanup.error && <div className="note bad">{cleanup.error}</div>}
      {tasks.loading && <Loading what="任务" />}
      {tasks.error && <Failed error={tasks.error} />}
      {tasks.data && tasks.data.length === 0 && <p className="muted">还没有任务。</p>}

      {tasks.data && tasks.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>阶段</th>
              <th>状态</th>
              <th>进度</th>
              <th>参数</th>
              <th>开始</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {tasks.data.map((task) => (
              <TaskRow
                key={task.id}
                task={task}
                open={openTask === task.id}
                onToggle={() => setOpenTask(openTask === task.id ? null : task.id)}
                onChanged={tasks.reload}
              />
            ))}
          </tbody>
        </table>
      )}

      {openTask !== null && <TaskLog taskId={openTask} onClose={() => setOpenTask(null)} />}
    </>
  )
}

function TaskRow({
  task,
  open,
  onToggle,
  onChanged,
}: {
  task: Task
  open: boolean
  onToggle: () => void
  onChanged: () => void
}) {
  const action = useAction<unknown>()
  const running = task.status === 'running' || task.status === 'queued'
  const ratio =
    task.status === 'succeeded' ? 1 : task.progress_total && task.progress_total > 0
      ? task.progress_done / task.progress_total
      : null

  return (
    <tr className={open ? 'selected' : ''}>
      <td className="mono">{task.id}</td>
      <td>{task.stage === 'verify' ? '链路测试' : task.stage}</td>
      <td>
        <span className={STATUS_CLASS[task.status]}>{STATUS_TEXT[task.status]}</span>
      </td>
      <td style={{ minWidth: 120 }}>
        {ratio !== null ? (
          <>
            <Bar value={ratio} kind={task.status === 'failed' ? 'bad' : undefined} />
            <span className="small mono muted">
              {task.status === 'succeeded' || task.progress_total === 10000
                ? `${Math.floor(ratio * 100)}%`
                : `${task.progress_done}/${task.progress_total}`}
            </span>
          </>
        ) : (
          <span className="small mono muted">{task.progress_done || '—'}</span>
        )}
        {task.progress_note && <span className="small muted"> {task.status === 'succeeded' ? '已完成' : task.progress_note}</span>}
      </td>
      <td className="small mono muted truncate" title={JSON.stringify(task.args)}>
        {Object.entries(task.args)
          .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join('+') : value}`)
          .join(' ') || '—'}
      </td>
      <td className="small muted">{task.started_at ?? '—'}</td>
      <td>
        <div className="row tight">
          <button className="action small" onClick={onToggle}>
            {open ? '收起' : '日志'}
          </button>
          {running ? (
            <button
              className="action small danger"
              disabled={action.busy}
              onClick={() =>
                action.run(async () => {
                  const result = await api.cancelTask(task.id)
                  onChanged()
                  return result
                })
              }
              title={task.stage === 'verify' ? '停止验证，保留已生成的层与远端 Space' : '停止是可续跑的：重新起同一阶段会接着上次的进度'}
            >
              停止
            </button>
          ) : (
            <button
              className="action small danger"
              disabled={action.busy}
              onClick={() =>
                action.run(async () => {
                  const result = await api.deleteTask(task.id)
                  onChanged()
                  return result
                })
              }
            >
              清理
            </button>
          )}
        </div>
        {action.error && <div className="small" style={{ color: 'var(--bad)' }}>{action.error}</div>}
      </td>
    </tr>
  )
}

/** 增量日志。只拉新增的事件，长任务的日志不会重复传。 */
function TaskLog({ taskId, onClose }: { taskId: number; onClose: () => void }) {
  const [detail, setDetail] = useState<TaskDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    let lastId = 0
    let running = true

    const tick = () => {
      api
        .task(taskId, lastId)
        .then((next) => {
          if (!alive) return
          if (next.events.length > 0) {
            lastId = next.events[next.events.length - 1]?.id ?? lastId
          }
          running = next.status === 'running' || next.status === 'queued'
          setDetail((previous) =>
            previous
              ? { ...next, events: [...previous.events, ...next.events].slice(-500) }
              : next,
          )
        })
        .catch((exc: unknown) => {
          if (alive) setError(String(exc instanceof Error ? exc.message : exc))
        })
    }

    tick()
    const timer = setInterval(() => running && tick(), 2000)
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
          任务 #{detail.id} · {detail.stage}{' '}
          <span className={STATUS_CLASS[detail.status]}>{STATUS_TEXT[detail.status]}</span>
        </h3>
        <div className="row tight small muted">
          <span>pid {detail.pid ?? '—'}</span>
          <span>退出码 {detail.exit_code ?? '—'}</span>
          <button className="action small" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>

      {detail.error && <div className="note bad">{detail.error}</div>}
      {detail.stage === 'ingest' && detail.status === 'running' && (
        <div className="note plain small">
          入库的进度条本质是「帮我盯着别人干活」—— 真正在编译的是 Akasha 的
          BullMQ worker，约 40 秒/篇是那边的吞吐，客户端调不动。停止只停客户端进程，
          服务端的编译不会因此停。
        </div>
      )}

      <pre className="block tall" style={{ marginTop: 10 }}>
        {detail.events.map((event) => event.message).join('\n') || '（还没有输出）'}
      </pre>
      <p className="small muted mono">{detail.argv.join(' ')}</p>
      <p className="small muted">
        参数在库里（run_config），不在命令行 —— 所以 argv 只有一个 id：
        {JSON.stringify(detail.args)}
      </p>
    </div>
  )
}
