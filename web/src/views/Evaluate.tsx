import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CompileRun, EvalDetail, MetricsView, Provider, QueryRun } from '../types'
import {
  CleanupButton,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pass,
  StatusTag,
  Timing,
  duration,
  metric,
  percent,
  useAction,
  useAsync,
  usePoll,
} from '../ui'

// 指标分族，顺序即勾选面板里的顺序，key 对应后端 registry 的 family。
const FAMILIES = [
  { key: 'retrieval', label: '检索质量', hint: '召回够不够，需要 gold 标注' },
  { key: 'qa', label: '生成质量', hint: '与参考答案比字面，确定性计算' },
  { key: 'attribution', label: '引用归因', hint: '引用与召回的差距' },
  { key: 'multihop', label: '多跳', hint: '图扩展的增量价值' },
  { key: 'judge', label: 'Judge 模型', hint: '逐条调模型' },
] as const

/** 评测层：针对某次查询结果配置评估参数并计算指标。 */
export function Evaluate({
  activeQuery,
  activeEval,
  onSelectQuery,
  onSelectEval,
  onAttribute,
  onOpenSettings,
  onOpenTasks,
}: {
  activeQuery: number | null
  activeEval: number | null
  onSelectQuery: (id: number) => void
  onSelectEval: (id: number | null) => void
  onAttribute: (evalId: number) => void
  onOpenSettings: () => void
  onOpenTasks: () => void
}) {
  const compiles = useAsync(() => api.compiles(), [])
  const cleanup = useAction<unknown>()

  const queries: QueryRun[] = (compiles.data?.compiles ?? []).flatMap((c: CompileRun) => c.queries)
  const done = queries.filter((q) => q.status === 'succeeded')
  const query = done.find((q) => q.id === activeQuery) ?? done[0] ?? null

  usePoll(
    queries.some((q) => q.evals.some((e) => e.status === 'running')),
    compiles.reload,
  )

  if (compiles.loading) return <Loading what="评测记录" />
  if (compiles.error) return <Failed error={compiles.error} />

  return (
    <>
      <h2>评测层</h2>

      {cleanup.error && <Failed error={cleanup.error} />}

      {done.length === 0 ? (
        <div className="note warn">还没有已完成的查询记录。</div>
      ) : (
        <div className="panel">
          <Field label="选择查询">
            <select
              value={query?.id ?? ''}
              onChange={(e) => {
                onSelectQuery(Number(e.target.value))
                onSelectEval(null)
              }}
            >
              {done.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name}（#{run.id}）
                </option>
              ))}
            </select>
          </Field>
          {query && (
            <NewEval
              query={query}
              onOpenSettings={onOpenSettings}
              onStarted={() => {
                compiles.reload()
                onOpenTasks()
              }}
            />
          )}
        </div>
      )}

      {query && query.evals.length > 0 && (
        <>
          <h3>评测记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>状态</th>
                <th>k</th>
                <th className="num">指标数</th>
                <th>耗时</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {query.evals.map((run) => (
                <tr key={run.id} className={run.id === activeEval ? 'selected' : ''}>
                  <td className="mono small">
                    {run.name} <span className="muted">#{run.id}</span>
                  </td>
                  <td>
                    <StatusTag status={run.status} />
                  </td>
                  <td className="small mono">{run.ks.join(', ')}</td>
                  <td className="num">{run.metrics.length}</td>
                  <td>
                    <Timing startedAt={run.created_at} finishedAt={run.finished_at} />
                  </td>
                  <td className="table-actions-cell">
                    <div className="table-actions">
                      <button
                        className="action small"
                        onClick={() => onSelectEval(run.id === activeEval ? null : run.id)}
                      >
                        {run.id === activeEval ? '收起' : '看结果'}
                      </button>
                      <button
                        className="action small"
                        disabled={run.status !== 'succeeded'}
                        onClick={() => onAttribute(run.id)}
                      >
                        去归因
                      </button>
                      <CleanupButton
                        what={`评测 ${run.name}`}
                        detail="指标、汇总、judge 判决与它下面的归因都会从数据库删除。"
                        busy={cleanup.busy}
                        onConfirm={() =>
                          cleanup.run(async () => {
                            const result = await api.deleteEval(run.id)
                            onSelectEval(null)
                            compiles.reload()
                            return result
                          })
                        }
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {activeEval !== null && <Results key={activeEval} evalId={activeEval} />}
    </>
  )
}

function NewEval({
  query,
  onStarted,
  onOpenSettings,
}: {
  query: QueryRun
  onStarted: () => void
  onOpenSettings: () => void
}) {
  const datasets = Object.keys(query.stats)
  const metrics = useAsync<MetricsView>(() => api.metrics(datasets), [datasets.join(',')])
  const providers = useAsync<Provider[]>(() => api.providers('judge'), [])
  const [selected, setSelected] = useState<string[]>([])
  const [name, setName] = useState('')
  const [ks, setKs] = useState('2,5,10')
  const [providerId, setProviderId] = useState<number | ''>('')
  const start = useAction<unknown>()

  // 默认勾选每一组都算得出来的那些。judge 逐条调模型，不默认勾。
  useEffect(() => {
    if (metrics.data && selected.length === 0) {
      const judge = new Set(
        metrics.data.definitions.filter((d) => d.kind === 'judge').map((d) => d.name),
      )
      setSelected(metrics.data.computable_for_all.filter((n) => !judge.has(n)))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metrics.data])

  if (metrics.loading) return <Loading what="指标清单" />
  if (metrics.error) return <Failed error={metrics.error} />
  if (!metrics.data) return null

  const judgeSelected = selected.some(
    (name) => metrics.data?.definitions.find((d) => d.name === name)?.kind === 'judge',
  )
  const partial = new Set(
    metrics.data.computable_for_some.filter((n) => !metrics.data!.computable_for_all.includes(n)),
  )

  return (
    <div style={{ marginTop: 12 }}>
      {start.error && <Failed error={start.error} />}

      <h4>指标</h4>
      {FAMILIES.map(({ key, label, hint }) => {
        const group = metrics.data!.definitions.filter((d) => d.family === key)
        if (group.length === 0) return null
        const usable = group.filter((d) => metrics.data!.computable_for_some.includes(d.name))
        const allOn = usable.length > 0 && usable.every((d) => selected.includes(d.name))

        return (
          <div key={key} className="metric-family">
            <div className="spread">
              <span className="small">
                <strong>{label}</strong>
                <span className="muted"> · {hint}</span>
              </span>
              {usable.length > 0 && (
                <button
                  className="action small"
                  onClick={() =>
                    setSelected(
                      allOn
                        ? selected.filter((n) => !usable.some((d) => d.name === n))
                        : [...new Set([...selected, ...usable.map((d) => d.name)])],
                    )
                  }
                >
                  {allOn ? '全不选' : '全选'}
                </button>
              )}
            </div>
            <div className="metric-grid">
              {group.map((definition) => {
                const computable = metrics.data!.computable_for_some.includes(definition.name)
                return (
                  <label
                    key={definition.name}
                    className={`check${computable ? '' : ' unavailable'}`}
                    title={definition.description}
                  >
                    <input
                      type="checkbox"
                      disabled={!computable}
                      checked={selected.includes(definition.name)}
                      onChange={() =>
                        setSelected(
                          selected.includes(definition.name)
                            ? selected.filter((n) => n !== definition.name)
                            : [...selected, definition.name],
                        )
                      }
                    />
                    {definition.name}
                    {definition.kind === 'judge' && <span className="tag warn">模型</span>}
                    {partial.has(definition.name) && <span className="tag">部分</span>}
                  </label>
                )
              })}
            </div>
          </div>
        )
      })}

      <div className="row" style={{ marginTop: 10 }}>
        <Field label="评测名称" hint="留空自动生成">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="自动" />
        </Field>
        <Field label="k 值" hint="逗号分隔">
          <input value={ks} onChange={(e) => setKs(e.target.value)} />
        </Field>
        {judgeSelected && (
          <Field label="评估模型 Judge">
            <select
              value={providerId}
              onChange={(e) => setProviderId(e.target.value === '' ? '' : Number(e.target.value))}
            >
              <option value="">最近配置的</option>
              {(providers.data ?? []).map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.label} · {provider.model}
                </option>
              ))}
            </select>
          </Field>
        )}
      </div>

      {judgeSelected && (providers.data ?? []).length === 0 && (
        <div className="note warn">
          勾了 Judge 指标但还没配 judge 模型端点。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenSettings}>
            去配置
          </button>
        </div>
      )}

      <div className="panel-actions">
        <button
          className="action primary"
          disabled={start.busy || selected.length === 0}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('evaluate', {
                query_id: query.id,
                metrics: selected,
                ks: ks
                  .split(',')
                  .map((k) => Number(k.trim()))
                  .filter((k) => k > 0),
                ...(name.trim() ? { name: name.trim() } : {}),
                ...(providerId === '' ? {} : { judge_provider_id: providerId }),
              })
              onStarted()
              return task
            })
          }
        >
          {start.busy ? '启动中…' : '开始评测'}
        </button>
        <span className="small muted">已选 {selected.length} 个指标</span>
      </div>
    </div>
  )
}

function Results({ evalId }: { evalId: number }) {
  const detail = useAsync<EvalDetail>(() => api.evalRun(evalId), [evalId])
  const [dataset, setDataset] = useState<string>('')

  if (detail.loading) return <Loading what="评测结果" />
  if (detail.error) return <Failed error={detail.error} />
  if (!detail.data) return null

  const data = detail.data
  const shown = dataset ? data.datasets.filter((d) => d.dataset === dataset) : data.datasets

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>
          {data.name} <span className="muted small">#{data.id}</span>
        </h3>
        <div className="row tight">
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="">全部数据集</option>
            {data.datasets.map((entry) => (
              <option key={entry.dataset} value={entry.dataset}>
                {entry.dataset}
              </option>
            ))}
          </select>
        </div>
      </div>

      {data.judge.total > 0 && (
        <div className={`note${data.judge.failure_rate > 0.1 ? ' bad' : ' plain'}`}>
          {/* 跨全部 judge 指标的合计，逐指标的均值在下面那张表里。 */}
          <strong>Judge 调用 {data.judge.total} 次</strong> · 已评分 {data.judge.scored}，失败率{' '}
          {percent(data.judge.failure_rate)}
          {data.judge.latency_mean !== null && (
            <> · 平均 {duration(data.judge.latency_mean)}/条</>
          )}
          。失败的样本被排除而不是记 0 —— 记 0 会让限流伪装成质量差。
          {Object.keys(data.judge.failures_by_kind).length > 0 && (
            <div className="small mono">{JSON.stringify(data.judge.failures_by_kind)}</div>
          )}
        </div>
      )}

      {shown.map((entry) => (
        <div key={entry.dataset} style={{ marginBottom: 18 }}>
          <h4>
            {entry.dataset} · 样本 {entry.responses_evaluated}
            {entry.http_failures > 0 && (
              <span className="tag bad" style={{ marginLeft: 6 }}>
                HTTP 失败 {entry.http_failures}
              </span>
            )}
          </h4>

          <p className="small muted">
            回答模式：
            {Object.entries(entry.answer_modes).map(([mode, share]) => (
              <span key={mode} style={{ marginLeft: 6 }}>
                <ModeTag mode={mode} /> {percent(share)}
              </span>
            ))}
          </p>

          {entry.omitted_metrics.length > 0 && (
            <div className="note warn small">
              省略了 {entry.omitted_metrics.length} 个指标：{entry.dataset} 没有 gold 文档标注。
            </div>
          )}

          <ScopeTable scopes={entry.scopes} />
        </div>
      ))}
    </div>
  )
}

/** 全样本 vs 仅 knowledge 并排，差值即生成端拒答的规模。 */
function ScopeTable({ scopes }: { scopes: Record<string, Record<string, number>> }) {
  const names = Array.from(
    new Set(Object.values(scopes).flatMap((entry) => Object.keys(entry))),
  ).sort()
  if (names.length === 0) return <p className="small muted">没有指标值。</p>

  return (
    <table>
      <thead>
        <tr>
          <th>指标</th>
          <th className="num">全样本</th>
          <th className="num">仅 knowledge</th>
          <th className="num">judge</th>
        </tr>
      </thead>
      <tbody>
        {names.map((name) => (
          <tr key={name}>
            <td className="mono small">{name}</td>
            <td className="num">{metric(scopes.overall ?? {}, name)}</td>
            <td className="num">{metric(scopes.knowledge_only ?? {}, name)}</td>
            <td className="num">{metric(scopes.judge ?? {}, name)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export { Pass }
