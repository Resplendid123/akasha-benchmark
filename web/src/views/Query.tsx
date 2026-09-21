import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CompileRun, QueryRun } from '../types'
import {
  AnswerModeFilter,
  CleanupButton,
  ConfigPanel,
  DatasetPicker,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pager,
  RecordNav,
  RecordSearch,
  StatusTag,
  Timing,
  num,
  useAction,
  useAsync,
  usePagedRecordNavigation,
  usePoll,
} from '../ui'

/** 查询层：选一次编译的空间，配置查询参数，逐条跑 query。 */
export function Query({
  activeCompile,
  activeQuery,
  onSelectCompile,
  onOpenCompile,
  onEvaluate,
  onOpenSettings,
  onOpenTasks,
}: {
  activeCompile: number | null
  activeQuery: number | null
  onSelectCompile: (id: number) => void
  onOpenCompile: (compileId: number) => void
  onEvaluate: (queryId: number) => void
  onOpenSettings: () => void
  onOpenTasks: () => void
}) {
  const compiles = useAsync(() => api.compiles(), [])
  const [openQuery, setOpenQuery] = useState<number | null>(activeQuery)
  const cleanup = useAction<unknown>()
  const retry = useAction<unknown>()

  const runs = compiles.data?.compiles ?? []
  useEffect(() => {
    if (activeQuery !== null) setOpenQuery(activeQuery)
  }, [activeQuery])
  usePoll(
    runs.some((c) => c.queries.some((q) => q.status === 'running')),
    compiles.reload,
  )

  if (compiles.loading) return <Loading what="查询记录" />
  if (compiles.error) return <Failed error={compiles.error} />
  if (!compiles.data) return null

  const ready = runs.filter((c) => c.readiness.ready)
  const compile = ready.find((c) => c.id === activeCompile) ?? ready[0] ?? null

  return (
    <>
      <h2>查询层</h2>

      {cleanup.error && <Failed error={cleanup.error} />}
      {retry.error && <Failed error={retry.error} />}

      {ready.length === 0 ? (
        <div className="note warn">
          还没有可用于查询的编译。编译需要成功结束、语料全部导入、且质量闸门通过。
        </div>
      ) : (
        <ConfigPanel storageKey="query" title="查询配置">
          <Field label="选择编译">
            <select
              value={compile?.id ?? ''}
              onChange={(e) => onSelectCompile(Number(e.target.value))}
            >
              {ready.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.run_id}（#{run.id}，{run.datasets.join('+')}
                  {run.readiness.warnings.length ? '，部分可用' : ''}）
                </option>
              ))}
            </select>
          </Field>
          {compile && (
            <NewQuery
              compile={compile}
              onOpenSettings={onOpenSettings}
              onStarted={() => {
                compiles.reload()
                onOpenTasks()
              }}
            />
          )}
        </ConfigPanel>
      )}

      {compile && compile.queries.length > 0 && (
        <>
          <h3>查询记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>Answer 模型</th>
                <th>状态</th>
                <th className="num">样本数</th>
                <th className="num">成功</th>
                <th className="num">并发</th>
                <th>耗时</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {compile.queries.map((run) => {
                const stats = Object.values(run.stats)
                const failures = stats.reduce((sum, entry) => sum + entry.failures, 0)
                const mean = stats.length
                  ? stats.reduce((sum, s) => sum + (s.latency_mean ?? 0), 0) / stats.length
                  : null
                return (
                  <tr key={run.id} className={openQuery === run.id ? 'selected' : ''}>
                  <td className="mono small">
                      {run.name} <span className="muted">#{run.id}</span>
                  </td>
                  <td className="small">{run.config_group ?? '—'}</td>
                    <td>
                      <StatusTag status={run.status} />
                    </td>
                  <td className="num">{run.sample_count}</td>
                  <td className="num">{run.success_count}</td>
                  <td className="num mono">{run.concurrency}</td>
                    <td>
                      <Timing
                        startedAt={run.created_at}
                        finishedAt={run.finished_at}
                        latencyMs={mean}
                        perLabel="条"
                      />
                    </td>
                    <td className="table-actions-cell">
                      <div className="table-actions">
                        <button
                          className="action small"
                          onClick={() => setOpenQuery(openQuery === run.id ? null : run.id)}
                        >
                          {openQuery === run.id ? '收起' : '响应'}
                        </button>
                        <button
                          className="action small"
                          disabled={run.status !== 'succeeded'}
                          onClick={() => onEvaluate(run.id)}
                        >
                          去评测
                        </button>
                        <button className="action small" onClick={() => onOpenCompile(run.compile_id)}>
                          回到编译
                        </button>
                        {failures > 0 && (
                          <button
                            className="action small primary"
                            disabled={retry.busy}
                            onClick={() =>
                              retry.run(async () => {
                                const task = await api.retryFailedQuery(run.id)
                                compiles.reload()
                                onOpenTasks()
                                return task
                              })
                            }
                          >
                            重试失败（{failures}）
                          </button>
                        )}
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

function NewQuery({
  compile,
  onStarted,
  onOpenSettings,
}: {
  compile: CompileRun
  onStarted: () => void
  onOpenSettings: () => void
}) {
  const models = useAsync(() => api.akashaModels('answer'), [])
  const [selected, setSelected] = useState<string[]>(compile.datasets)
  const [name, setName] = useState('')
  const [limit, setLimit] = useState<number | ''>('')
  const [threshold, setThreshold] = useState<number | ''>('')
  const [concurrency, setConcurrency] = useState(1)
  const [answerModelId, setAnswerModelId] = useState<number | ''>('')
  const start = useAction<unknown>()

  const available = Object.keys(compile.stats)
  const answerModels = models.data?.models ?? []

  useEffect(() => {
    if (answerModels.length === 0) {
      setAnswerModelId('')
      return
    }
    if (!answerModels.some((entry) => entry.id === answerModelId)) {
      setAnswerModelId(answerModels[0]!.id)
    }
  }, [answerModels, answerModelId])

  if (models.loading) return <Loading what="Answer 模型配置" />
  if (models.error) return <Failed error={models.error} />

  return (
    <div style={{ marginTop: 12 }}>
      {start.error && <Failed error={start.error} />}
      {compile.readiness.warnings.map((warning) => (
        <div key={warning} className="note warn">
          {warning}
        </div>
      ))}
      <DatasetPicker
        all={available}
        selected={selected.filter((n) => available.includes(n))}
        onChange={setSelected}
      />
      <div className="row" style={{ marginTop: 10 }}>
        <Field label="查询名称" hint="留空自动生成">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="自动" />
        </Field>
        <Field label="Answer 模型" hint="选择即在查询启动时应用到 Akasha">
          <select
            value={answerModelId}
            onChange={(event) =>
              setAnswerModelId(event.target.value === '' ? '' : Number(event.target.value))
            }
          >
            <option value="">未选择</option>
            {answerModels.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label} · {entry.model}
              </option>
            ))}
          </select>
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
        <Field label="并发" hint="并发查询">
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
          disabled={start.busy || selected.length === 0 || answerModelId === ''}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('query', {
                compile_id: compile.id,
                datasets: selected,
                concurrency,
                answer_model_id: answerModelId,
                ...(name.trim() ? { name: name.trim() } : {}),
                ...(limit === '' ? {} : { sample_limit: limit }),
                ...(threshold === '' ? {} : { score_threshold: threshold }),
              })
              onStarted()
              return task
            })
          }
        >
          {start.busy ? '应用模型并启动中…' : '应用模型并开始查询'}
        </button>
      </div>
      {answerModels.length === 0 && (
        <div className="note warn">
          还没有 Answer 模型配置。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenSettings}>
            去设置
          </button>
        </div>
      )}
    </div>
  )
}

function Responses({ queryId }: { queryId: number }) {
  const [mode, setMode] = useState('')
  const [offset, setOffset] = useState(0)
  const [sampleId, setSampleId] = useState<string | null>(null)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const limit = 5

  const { data, error, loading } = useAsync(
    () => api.responses(queryId, { answer_mode: mode || undefined, q, limit, offset }),
    [queryId, mode, q, offset],
  )
  const nav = usePagedRecordNavigation({
    items: data?.responses ?? [],
    total: data?.total ?? 0,
    responseOffset: data?.offset ?? offset,
    offset,
    limit,
    selectedKey: sampleId,
    itemKey: (row) => row.sample_id,
    onSelect: (row) => setSampleId(row.sample_id),
    onOffsetChange: setOffset,
  })

  // 完整响应覆盖整个列表框，带返回按钮。
  if (sampleId) {
    return (
      <div className="panel" style={{ marginTop: 14 }}>
        <div className="spread">
          <h3 style={{ margin: 0 }}>
            查询 #{queryId} · 样本 {sampleId}
          </h3>
          <RecordNav
            hasPrevious={nav.hasPrevious}
            hasNext={nav.hasNext}
            onPrevious={nav.previous}
            onNext={nav.next}
            onBack={() => setSampleId(null)}
            backLabel="返回响应列表"
            position={nav.position}
            total={data?.total ?? 0}
            busy={loading || nav.navigating}
          />
        </div>
        <FullResponse key={sampleId} queryId={queryId} sampleId={sampleId} />
      </div>
    )
  }

  if (loading) return <Loading what="响应" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="record-filters">
        <AnswerModeFilter
          counts={data.count_by_answer_mode}
          value={mode}
          onChange={(value) => {
            setMode(value)
            setOffset(0)
            setSampleId(null)
          }}
        />
        <RecordSearch
          placeholder="搜索 sample_id、问题或系统答案"
          value={term}
          onChange={setTerm}
          onSearch={(value) => {
            setQ(value)
            setOffset(0)
            setSampleId(null)
          }}
        />
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
                <button className="action small" onClick={() => setSampleId(row.sample_id)}>
                  完整响应
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
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
