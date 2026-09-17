import { useEffect, useState } from 'react'
import { api } from '../api'
import type {
  CompileRun,
  EvalDetail,
  EvalSampleDetail,
  EvalSampleList,
  MetricsView,
  Provider,
  QueryRun,
} from '../types'
import {
  AnswerModeFilter,
  CleanupButton,
  ConfigPanel,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pager,
  Pass,
  RecordSearch,
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
        <ConfigPanel storageKey="evaluate" title="评测配置">
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
        </ConfigPanel>
      )}

      {query && query.evals.length > 0 && (
        <>
          <h3>评测记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>配置组</th>
                <th>状态</th>
                <th className="num">样本数</th>
                <th className="num">成功</th>
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
                  <td className="small">{run.config_group ?? '—'}</td>
                  <td>
                    <StatusTag status={run.status} />
                  </td>
                  <td className="num">{run.sample_count}</td>
                  <td className="num">{run.success_count}</td>
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

  useEffect(() => {
    const latest = providers.data?.[0]
    if (latest && !(providers.data ?? []).some((provider) => provider.id === providerId)) {
      setProviderId(latest.id)
    }
  }, [providers.data, providerId])

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
      <div className="row tight" style={{ marginBottom: 8 }}>
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <option value="">全部数据集</option>
          {data.datasets.map((entry) => (
            <option key={entry.dataset} value={entry.dataset}>
              {entry.dataset}
            </option>
          ))}
        </select>
      </div>

      {data.judge.total > 0 && (
        <div className={`note${data.judge.failure_rate > 0.1 ? ' bad' : ' plain'}`}>
          {/* 跨全部 judge 指标的合计，逐指标的均值在下面那张表里。 */}
          <strong>Judge 调用 {data.judge.total} 次</strong> · 已评分 {data.judge.scored}，失败率{' '}
          {percent(data.judge.failure_rate)}
          {data.judge.latency_mean !== null && (
            <> · 平均 {duration(data.judge.latency_mean)}/条</>
          )}
        </div>
      )}

      {shown.map((entry) => (
        <div key={entry.dataset} style={{ marginBottom: 18 }}>
          {entry.http_failures > 0 && (
            <span className="tag bad">HTTP 失败 {entry.http_failures}</span>
          )}

          <p className="small muted">
            回答模式：
            {Object.entries(entry.answer_modes).map(([mode, share]) => (
              <span key={mode} style={{ marginLeft: 6 }}>
                <ModeTag mode={mode} /> {percent(share)}
              </span>
            ))}
          </p>

          <ScopeTable scopes={entry.scopes} />
        </div>
      ))}

      <SampleResults evalId={evalId} dataset={dataset} />
    </div>
  )
}

function SampleResults({ evalId, dataset }: { evalId: number; dataset: string }) {
  const [offset, setOffset] = useState(0)
  const [openSample, setOpenSample] = useState<string | null>(null)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const [mode, setMode] = useState('')
  const limit = 5
  const samples = useAsync<EvalSampleList>(
    () =>
      api.evalSamples(evalId, {
        dataset: dataset || undefined,
        answer_mode: mode || undefined,
        q,
        limit,
        offset,
      }),
    [evalId, dataset, mode, q, offset],
  )

  useEffect(() => {
    setOffset(0)
    setOpenSample(null)
    setTerm('')
    setQ('')
    setMode('')
  }, [dataset])

  // 样本详情覆盖整个列表，带返回按钮。
  if (openSample) {
    return (
      <div style={{ marginTop: 18 }}>
        <div className="spread">
          <h4 style={{ margin: 0 }}>样本 {openSample}</h4>
          <button className="action small" onClick={() => setOpenSample(null)}>
            ← 返回样本列表
          </button>
        </div>
        <EvalSampleView key={openSample} evalId={evalId} sampleId={openSample} />
      </div>
    )
  }

  if (samples.loading) return <Loading what="评测样本" />
  if (samples.error) return <Failed error={samples.error} />
  if (!samples.data) return null

  return (
    <div style={{ marginTop: 18 }}>
      <div className="record-filters">
        <AnswerModeFilter
          counts={samples.data.count_by_answer_mode}
          value={mode}
          onChange={(value) => {
            setMode(value)
            setOffset(0)
            setOpenSample(null)
          }}
        />
        <RecordSearch
          placeholder="搜索 sample_id、问题或系统答案"
          value={term}
          onChange={setTerm}
          onSearch={(value) => {
            setQ(value)
            setOffset(0)
          }}
        />
      </div>
      {samples.data.total === 0 ? (
        <p className="small muted">没有匹配的样本。</p>
      ) : (
      <table>
        <thead>
          <tr>
            <th>sample_id</th>
            <th>问题</th>
            <th>答案模式</th>
            <th>指标</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {samples.data.samples.map((sample) => (
            <tr key={sample.sample_id}>
              <td className="mono small">{sample.sample_id}</td>
              <td className="small">{sample.question}</td>
              <td><ModeTag mode={sample.answer_mode} /></td>
              <td>
                <MetricTags metrics={sample.metrics} />
              </td>
              <td>
                <button className="action small" onClick={() => setOpenSample(sample.sample_id)}>
                  查看详情
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      )}
      {samples.data.total > 0 && (
        <Pager
          total={samples.data.total}
          offset={offset}
          limit={limit}
          onChange={setOffset}
        />
      )}
    </div>
  )
}

function EvalSampleView({ evalId, sampleId }: { evalId: number; sampleId: string }) {
  const detail = useAsync<EvalSampleDetail>(() => api.evalSample(evalId, sampleId), [evalId, sampleId])
  if (detail.loading) return <Loading what="样本明细" />
  if (detail.error) return <Failed error={detail.error} />
  if (!detail.data) return null
  const response = detail.data.response ?? {}
  const retrieved = records(response.retrievedSources)
  const snippets = records(response.snippets)
  const citations = records(response.citations)
  const evidence = records(response.citationEvidence)
  return (
    <div className="panel flat" style={{ marginTop: 10 }}>
      <h4>响应概览</h4>
      <dl className="kv">
        <dt>问题</dt><dd>{text(detail.data.detail.question)}</dd>
        <dt>参考答案</dt>
        <dd>{strings(detail.data.detail.reference_answers).join(' / ') || '—'}</dd>
        <dt>回答模式</dt><dd><ModeTag mode={detail.data.answer_mode} /></dd>
        <dt>HTTP 状态</dt>
        <dd><span className={`tag ${detail.data.http_status >= 200 && detail.data.http_status < 300 ? 'ok' : 'bad'}`}>{detail.data.http_status}</span></dd>
        <dt>系统答案</dt><dd>{detail.data.answer || '—'}</dd>
      </dl>

      <h4>逐样本指标</h4>
      <MetricTags metrics={detail.data.metrics} />

      <RetrievalResults snippets={snippets} retrieved={retrieved} />

      <h4>引用证据（{citations.length}）</h4>
      {citations.length === 0 ? (
        <p className="small muted">没有引用。</p>
      ) : (
        <div className="detail-list">
          {citations.map((citation, index) => (
            <DetailCard key={index} title={`引用 ${index + 1}`}>
              <MetaLine record={citation} fields={[['sourcePageId', 'page_id'], ['title', '标题']]} />
              {strings(evidence[index]?.excerpts).map((excerpt, excerptIndex) => (
                <blockquote key={excerptIndex}>{excerpt}</blockquote>
              ))}
            </DetailCard>
          ))}
        </div>
      )}

      {detail.data.judge_verdicts.length > 0 && <h4>Judge 判据</h4>}
      {detail.data.judge_verdicts.map((verdict) => (
        <JudgeVerdictView key={verdict.metric} verdict={verdict} />
      ))}
    </div>
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function text(value: unknown, fallback = '—'): string {
  return typeof value === 'string' && value.trim() ? value : fallback
}

function MetricTags({ metrics }: { metrics: Record<string, number> }) {
  const entries = Object.entries(metrics)
  if (entries.length === 0) return <span className="small muted">没有指标值。</span>
  return (
    <div className="metric-tags">
      {entries.map(([name, value]) => (
        <span key={name} className="tag">{name} · {Number(value).toFixed(4)}</span>
      ))}
    </div>
  )
}

function RetrievalResults({
  snippets,
  retrieved,
}: {
  snippets: Record<string, unknown>[]
  retrieved: Record<string, unknown>[]
}) {
  const [offset, setOffset] = useState(0)
  const limit = 5
  const items = snippets.length > 0 ? snippets : retrieved
  const rich = snippets.length > 0

  return (
    <>
      <h4>检索结果（{items.length}）</h4>
      {items.length === 0 ? (
        <p className="small muted">没有检索来源。</p>
      ) : (
        <>
          <div className="detail-list">
            {items.slice(offset, offset + limit).map((item, index) => (
              <DetailCard
                key={offset + index}
                title={`${offset + index + 1}. ${text(item.title, '无标题')}`}
              >
                <MetaLine
                  record={item}
                  fields={
                    rich
                      ? [['score', '分数'], ['sourcePageId', 'page_id']]
                      : [['sourcePageId', 'page_id'], ['id', 'ID'], ['score', '分数']]
                  }
                />
                {rich && (
                  <>
                    <TagList values={strings(item.retrievalReasons)} />
                    <div className="readable-text">{text(item.text, '无正文')}</div>
                  </>
                )}
              </DetailCard>
            ))}
          </div>
          {items.length > limit && (
            <Pager
              total={items.length}
              offset={offset}
              limit={limit}
              onChange={setOffset}
            />
          )}
        </>
      )}
    </>
  )
}

function DetailCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="detail-card">
      <strong>{title}</strong>
      {children}
    </section>
  )
}

function MetaLine({
  record,
  fields,
}: {
  record: Record<string, unknown>
  fields: [string, string][]
}) {
  const shown = fields.filter(([key]) => record[key] !== null && record[key] !== undefined && record[key] !== '')
  if (shown.length === 0) return null
  return (
    <div className="detail-meta">
      {shown.map(([key, label]) => <span key={key}>{label}：<span className="mono">{String(record[key])}</span></span>)}
    </div>
  )
}

function TagList({ values }: { values: string[] }) {
  if (values.length === 0) return null
  return <div className="metric-tags">{values.map((value) => <span key={value} className="tag accent">{value}</span>)}</div>
}

function JudgeVerdictView({ verdict }: { verdict: EvalSampleDetail['judge_verdicts'][number] }) {
  const detail = verdict.detail ?? {}
  const claims = records(detail.claims)
  const sentences = records(detail.sentences)
  const passages = records(detail.passages)
  const hidden = new Set([
    'claims', 'sentences', 'passages', 'raw_response', 'raw_http_response',
    'claim_count', 'sentence_count', 'passage_count', 'supported', 'unsupported',
    'contradicted', 'relevant', 'useful', 'verdict', 'reason', 'skipped', 'error',
  ])
  const extras = Object.entries(detail).filter(([key]) => !hidden.has(key))
  return (
    <section className="judge-card">
      <div className="spread">
        <strong>{verdict.metric}</strong>
        <div className="row tight">
          <span className={`tag ${verdict.failure_kind ? 'bad' : verdict.score === null ? '' : 'ok'}`}>
            {verdict.failure_kind ?? (verdict.score === null ? '无定义' : verdict.score.toFixed(4))}
          </span>
          {verdict.latency_ms !== null && <span className="small muted">{duration(verdict.latency_ms)}</span>}
        </div>
      </div>
      {Boolean(detail.skipped) && <p className="small muted">已跳过：{String(detail.skipped)}</p>}
      {Boolean(detail.error) && <p className="small" style={{ color: 'var(--bad)' }}>{String(detail.error)}</p>}
      {Boolean(detail.verdict) && <p><strong>结论：</strong>{String(detail.verdict)}</p>}
      {Boolean(detail.reason) && <p><strong>理由：</strong>{String(detail.reason)}</p>}
      {claims.length > 0 && <VerdictItems items={claims} textKey="claim" textLabel="陈述" />}
      {sentences.length > 0 && <VerdictItems items={sentences} textKey="sentence" textLabel="句子" />}
      {passages.length > 0 && <VerdictItems items={passages} textKey="index" textLabel="段落" numericKey />}
      {extras.length > 0 && (
        <dl className="kv compact-kv">
          {extras.map(([key, value]) => (
            <div key={key} className="kv-row"><dt>{key}</dt><dd><ReadableValue value={value} /></dd></div>
          ))}
        </dl>
      )}
    </section>
  )
}

function VerdictItems({ items, textKey, textLabel, numericKey = false }: { items: Record<string, unknown>[]; textKey: string; textLabel: string; numericKey?: boolean }) {
  return (
    <div className="detail-list">
      {items.map((item, index) => (
        <div key={index} className="verdict-item">
          <div>
            <strong>{textLabel} {numericKey ? `#${String(item[textKey] ?? index + 1)}` : index + 1}：</strong>
            {!numericKey && String(item[textKey] ?? '')}
          </div>
          <div className="row tight">
            {item.verdict !== undefined && <span className="tag">{String(item.verdict)}</span>}
            {item.evidence !== undefined && item.evidence !== '' && <span className="small muted">证据：{String(item.evidence)}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function ReadableValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === '') return <>—</>
  if (Array.isArray(value)) return <>{value.map((item, index) => <div key={index}><ReadableValue value={item} /></div>)}</>
  if (isRecord(value)) return <dl className="kv compact-kv">{Object.entries(value).map(([key, item]) => <div key={key} className="kv-row"><dt>{key}</dt><dd><ReadableValue value={item} /></dd></div>)}</dl>
  return <>{String(value)}</>
}

/** 并排展示全样本与 knowledge 子集的指标均值。 */
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
