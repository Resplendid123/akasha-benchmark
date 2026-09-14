import { useState } from 'react'
import { api } from '../api'
import type { CompileRun, QueryRun } from '../types'
import {
  CleanupButton,
  DatasetPicker,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pager,
  StatusTag,
  Timing,
  num,
  useAction,
  useAsync,
  usePoll,
} from '../ui'

/** 查询层：选一次编译的空间，配置查询参数，逐条跑 query。 */
export function Query({
  activeCompile,
  onSelectCompile,
  onEvaluate,
  onOpenTasks,
}: {
  activeCompile: number | null
  onSelectCompile: (id: number) => void
  onEvaluate: (queryId: number) => void
  onOpenTasks: () => void
}) {
  const compiles = useAsync(() => api.compiles(), [])
  const [openQuery, setOpenQuery] = useState<number | null>(null)
  const cleanup = useAction<unknown>()

  const runs = compiles.data?.compiles ?? []
  usePoll(
    runs.some((c) => c.queries.some((q) => q.status === 'running')),
    compiles.reload,
  )

  if (compiles.loading) return <Loading what="查询记录" />
  if (compiles.error) return <Failed error={compiles.error} />
  if (!compiles.data) return null

  const ready = runs.filter((c) => c.readiness.ready)
  const compile = runs.find((c) => c.id === activeCompile) ?? ready[0] ?? null

  return (
    <>
      <h2>查询层</h2>

      {cleanup.error && <Failed error={cleanup.error} />}

      {ready.length === 0 ? (
        <div className="note warn">
          还没有可用于查询的编译。编译需要成功结束、语料全部导入、且质量闸门通过。
        </div>
      ) : (
        <div className="panel">
          <Field label="选择编译">
            <select
              value={compile?.id ?? ''}
              onChange={(e) => onSelectCompile(Number(e.target.value))}
            >
              {ready.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.run_id}（#{run.id}，{run.datasets.join('+')}）
                </option>
              ))}
            </select>
          </Field>
          {compile && (
            <NewQuery
              compile={compile}
              onStarted={() => {
                compiles.reload()
                onOpenTasks()
              }}
            />
          )}
        </div>
      )}

      {compile && compile.queries.length > 0 && (
        <>
          <h3>查询记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>状态</th>
                <th className="num">响应</th>
                <th className="num">失败</th>
                <th>耗时</th>
                <th>阈值</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {compile.queries.map((run) => {
                const stats = Object.values(run.stats)
                const responses = stats.reduce((sum, s) => sum + s.responses, 0)
                const failures = stats.reduce((sum, s) => sum + s.failures, 0)
                const mean = stats.length
                  ? stats.reduce((sum, s) => sum + (s.latency_mean ?? 0), 0) / stats.length
                  : null
                return (
                  <tr key={run.id} className={openQuery === run.id ? 'selected' : ''}>
                    <td className="mono small">
                      {run.name} <span className="muted">#{run.id}</span>
                    </td>
                    <td>
                      <StatusTag status={run.status} />
                    </td>
                    <td className="num">{responses}</td>
                    <td className="num">{failures || '—'}</td>
                    <td>
                      <Timing
                        startedAt={run.created_at}
                        finishedAt={run.finished_at}
                        latencyMs={mean}
                        perLabel="条"
                      />
                    </td>
                    <td className="small muted">{run.score_threshold ?? '服务端默认'}</td>
                    <td className="table-actions-cell">
                      <div className="table-actions">
                        <button
                          className="action small"
                          onClick={() => setOpenQuery(openQuery === run.id ? null : run.id)}
                        >
                          {openQuery === run.id ? '收起' : '看响应'}
                        </button>
                        <button
                          className="action small"
                          disabled={run.status !== 'succeeded'}
                          onClick={() => onEvaluate(run.id)}
                        >
                          去评测
                        </button>
                        <CleanupButton
                          what={`查询 ${run.name}`}
                          detail="响应与它下面的评测、归因都会从数据库删除。"
                          busy={cleanup.busy}
                          onConfirm={() =>
                            cleanup.run(async () => {
                              const result = await api.deleteQuery(run.id)
                              setOpenQuery(null)
                              compiles.reload()
                              return result
                            })
                          }
                        />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      )}

      {openQuery !== null && <Responses key={openQuery} queryId={openQuery} />}
    </>
  )
}

function NewQuery({ compile, onStarted }: { compile: CompileRun; onStarted: () => void }) {
  const [selected, setSelected] = useState<string[]>(compile.datasets)
  const [name, setName] = useState('')
  const [limit, setLimit] = useState<number | ''>('')
  const [threshold, setThreshold] = useState<number | ''>('')
  const [concurrency, setConcurrency] = useState(1)
  const start = useAction<unknown>()

  const available = Object.keys(compile.stats)

  return (
    <div style={{ marginTop: 12 }}>
      {start.error && <Failed error={start.error} />}
      <DatasetPicker
        all={available}
        selected={selected.filter((n) => available.includes(n))}
        onChange={setSelected}
      />
      <div className="row" style={{ marginTop: 10 }}>
        <Field label="查询名称" hint="留空自动生成；填已有的则续跑">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="自动" />
        </Field>
        <Field label="每组样本数" hint="留空则跑全部">
          <input
            type="number"
            min={1}
            value={limit}
            onChange={(e) => setLimit(e.target.value === '' ? '' : Number(e.target.value))}
            placeholder="全部"
          />
        </Field>
        <Field label="分数阈值" hint="留空用服务端默认">
          <input
            type="number"
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(e.target.value === '' ? '' : Number(e.target.value))}
            placeholder="默认"
          />
        </Field>
        <Field label="并发" hint="不影响结果，只影响快慢">
          <input
            type="number"
            min={1}
            max={8}
            value={concurrency}
            onChange={(e) => setConcurrency(Number(e.target.value))}
          />
        </Field>
      </div>

      <div className="panel-actions">
        <button
          className="action primary"
          disabled={start.busy || selected.length === 0}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('query', {
                compile_id: compile.id,
                datasets: selected,
                concurrency,
                ...(name.trim() ? { name: name.trim() } : {}),
                ...(limit === '' ? {} : { sample_limit: limit }),
                ...(threshold === '' ? {} : { score_threshold: threshold }),
              })
              onStarted()
              return task
            })
          }
        >
          {start.busy ? '启动中…' : '开始查询'}
        </button>
      </div>
    </div>
  )
}

function Responses({ queryId }: { queryId: number }) {
  const [mode, setMode] = useState('')
  const [offset, setOffset] = useState(0)
  const [sampleId, setSampleId] = useState<string | null>(null)
  const limit = 10

  const { data, error, loading } = useAsync(
    () => api.responses(queryId, { answer_mode: mode || undefined, limit, offset }),
    [queryId, mode, offset],
  )

  if (loading) return <Loading what="响应" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <h3 style={{ marginTop: 0 }}>查询 #{queryId} 的响应</h3>

      <div className="row tight" style={{ marginBottom: 8 }}>
        {Object.entries(data.count_by_answer_mode).map(([key, count]) => (
          <button
            key={key}
            className={`action small${mode === key ? ' primary' : ''}`}
            onClick={() => {
              setMode(mode === key ? '' : key)
              setOffset(0)
            }}
          >
            {key} · {count}
          </button>
        ))}
      </div>

      <table>
        <thead>
          <tr>
            <th>sample_id</th>
            <th>问题</th>
            <th>模式</th>
            <th className="num">召回</th>
            <th className="num">引用</th>
            <th className="num">延迟</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.responses.map((row) => (
            <tr key={row.sample_id} className={sampleId === row.sample_id ? 'selected' : ''}>
              <td className="mono small">{row.sample_id}</td>
              <td className="small truncate">{row.question}</td>
              <td>
                {200 <= row.http_status && row.http_status < 300 ? (
                  <ModeTag mode={row.answer_mode} />
                ) : (
                  <span className="tag bad" title={row.error ?? ''}>
                    HTTP {row.http_status}
                  </span>
                )}
              </td>
              <td className="num">{row.retrieved_count}</td>
              <td className="num">{row.citation_count}</td>
              <td className="num">{num(row.latency_ms)}</td>
              <td>
                <button
                  className="action small"
                  onClick={() => setSampleId(sampleId === row.sample_id ? null : row.sample_id)}
                >
                  {sampleId === row.sample_id ? '收起' : '完整响应'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />

      {sampleId && <FullResponse key={sampleId} queryId={queryId} sampleId={sampleId} />}
    </div>
  )
}

function FullResponse({ queryId, sampleId }: { queryId: number; sampleId: string }) {
  const { data, error, loading } = useAsync(
    () => api.response(queryId, sampleId),
    [queryId, sampleId],
  )
  if (loading) return <Loading what="完整响应" />
  if (error) return <Failed error={error} />
  return (
    <pre className="block tall" style={{ marginTop: 10 }}>
      {JSON.stringify(data, null, 2)}
    </pre>
  )
}

export type { QueryRun }
