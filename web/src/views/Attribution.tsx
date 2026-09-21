import { useEffect, useState } from 'react'
import { api } from '../api'
import type {
  AttributionResult,
  AttributionRun,
  CompileRun,
  EvalRun,
  EvalSampleDetail,
  Provider,
  Snippet,
} from '../types'
import {
  CauseTag,
  CleanupButton,
  ConfigPanel,
  Collapsible,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pager,
  RecordNav,
  StatusTag,
  Timing,
  useAction,
  useAsync,
  usePagedRecordNavigation,
  usePoll,
} from '../ui'
import { LineageView } from './Compile'

/** 归因层：针对某次 编译->查询->评测 链路推出根因。
 *
 * 规则判据不需要模型就能出分类，模型只补一段因果叙述。
 */
export function Attribution({
  activeEval,
  onSelectEval,
  onOpenEval,
  onOpenSettings,
  onOpenTasks,
}: {
  activeEval: number | null
  onSelectEval: (id: number) => void
  onOpenEval: (queryId: number, evalId: number) => void
  onOpenSettings: () => void
  onOpenTasks: () => void
}) {
  const compiles = useAsync(() => api.compiles(), [])
  const [open, setOpen] = useState<number | null>(null)
  const cleanup = useAction<unknown>()

  const evals: EvalRun[] = (compiles.data?.compiles ?? []).flatMap((c: CompileRun) =>
    c.queries.flatMap((q) => q.evals),
  )
  const done = evals.filter((e) => e.status === 'succeeded')
  const current = done.find((e) => e.id === activeEval) ?? done[0] ?? null

  usePoll(
    evals.some((e) => e.attributions.some((a) => a.status === 'running')),
    compiles.reload,
  )

  if (compiles.loading) return <Loading what="归因记录" />
  if (compiles.error) return <Failed error={compiles.error} />

  return (
    <>
      <h2>归因层</h2>

      {cleanup.error && <Failed error={cleanup.error} />}

      {done.length === 0 ? (
        <div className="note warn">还没有已完成的评测记录。</div>
      ) : (
        <ConfigPanel storageKey="attribution" title="归因配置">
          <Field label="选择评测">
            <select value={current?.id ?? ''} onChange={(e) => onSelectEval(Number(e.target.value))}>
              {done.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name}（#{run.id}）
                </option>
              ))}
            </select>
          </Field>
          {current && (
            <NewAttribution
              evalRun={current}
              onOpenSettings={onOpenSettings}
              onStarted={() => {
                compiles.reload()
                onOpenTasks()
              }}
            />
          )}
        </ConfigPanel>
      )}

      {current && current.attributions.length > 0 && (
        <>
          <h3>归因记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>归因模型</th>
                <th>状态</th>
                <th className="num">样本数</th>
                <th className="num">成功</th>
                <th className="num">并发</th>
                <th>模型归因</th>
                <th>耗时</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {current.attributions.map((run: AttributionRun) => (
                <tr key={run.id} className={open === run.id ? 'selected' : ''}>
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
                    <span className={`tag ${run.provider_id !== null ? 'ok' : ''}`}>
                      {run.provider_id !== null ? '使用' : '未使用'}
                    </span>
                  </td>
                  <td>
                    <Timing startedAt={run.created_at} finishedAt={run.finished_at} />
                  </td>
                  <td className="table-actions-cell">
                    <div className="table-actions">
                      <button
                        className="action small"
                        onClick={() => setOpen(open === run.id ? null : run.id)}
                      >
                        {open === run.id ? '收起' : '看结论'}
                      </button>
                      <button
                        className="action small"
                        onClick={() => onOpenEval(current.query_id, run.eval_id)}
                      >
                        回到评测
                      </button>
                      <CleanupButton
                        what={`归因 ${run.name}`}
                        detail="归因结论会从数据库删除。评测结果不受影响。"
                        busy={cleanup.busy}
                        onConfirm={() =>
                          cleanup.run(async () => {
                            const result = await api.deleteAttribution(run.id)
                            setOpen(null)
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

      {open !== null && current && (
        <Conclusions key={open} attributionId={open} evalId={current.id} />
      )}
    </>
  )
}

function NewAttribution({
  evalRun,
  onStarted,
  onOpenSettings,
}: {
  evalRun: EvalRun
  onStarted: () => void
  onOpenSettings: () => void
}) {
  const providers = useAsync<Provider[]>(() => api.providers('attribution'), [])
  const [name, setName] = useState('')
  const [useModel, setUseModel] = useState(false)
  const [providerId, setProviderId] = useState<number | ''>('')
  const [concurrency, setConcurrency] = useState(1)
  const start = useAction<unknown>()

  useEffect(() => {
    const latest = providers.data?.[0]
    if (latest && !(providers.data ?? []).some((provider) => provider.id === providerId)) {
      setProviderId(latest.id)
    }
  }, [providers.data, providerId])

  return (
    <div style={{ marginTop: 12 }}>
      {start.error && <Failed error={start.error} />}
      <div className="row">
        <Field label="归因名称" hint="留空自动生成">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="自动" />
        </Field>
        {useModel && (
          <>
            <Field label="归因模型">
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
            <Field label="并发" hint="归因模型调用并发数">
              <input type="number" min={1} max={16} value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} />
            </Field>
          </>
        )}
      </div>

      <label className="check" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={useModel} onChange={() => setUseModel(!useModel)} />
        用模型补因果叙述（每条一次调用）
      </label>

      {useModel && (providers.data ?? []).length === 0 && (
        <div className="note warn">
          还没配归因模型端点；不配也能跑，只是没有因果叙述。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenSettings}>
            去配置
          </button>
        </div>
      )}

      <div style={{ marginTop: 10 }}>
        <button
          className="action primary"
          disabled={start.busy}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('attribute', {
                eval_id: evalRun.id,
                use_model: useModel,
                ...(name.trim() ? { name: name.trim() } : {}),
                ...(providerId === '' ? {} : { provider_id: providerId }),
                ...(useModel ? { concurrency } : {}),
              })
              onStarted()
              return task
            })
          }
        >
          {start.busy ? '启动中…' : '开始归因'}
        </button>
      </div>
    </div>
  )
}

const PAGE = 5

/** 模型的归因结论。``disagreement`` 单独标红，它表示根因可能判错了。 */
function ModelVerdict({ result }: { result: AttributionResult }) {
  const model = (result.evidence?.model ?? {}) as {
    contributing_factors?: unknown
    disagreement?: unknown
    confidence?: unknown
  }
  const error = result.evidence?.model_error ? String(result.evidence.model_error) : ''
  const factors = Array.isArray(model.contributing_factors)
    ? model.contributing_factors.map(String)
    : []
  const disagreement = model.disagreement ? String(model.disagreement) : ''

  return (
    <div className={`note ${disagreement ? 'warn' : 'plain'}`} style={{ marginTop: 10 }}>
      <div className="spread">
        <strong className="small mono">{result.sample_id}</strong>
        {model.confidence !== undefined && model.confidence !== null && (
          <span className="small muted">置信度 {String(model.confidence)}</span>
        )}
      </div>

      {error && (
        <div className="small" style={{ color: 'var(--bad)', marginTop: 4 }}>
          模型调用失败：{error}。下面只有规则结论。
        </div>
      )}

      {result.narrative && <div style={{ marginTop: 4 }}>{result.narrative}</div>}

      {disagreement && (
        <div style={{ marginTop: 6 }}>
          <span className="tag warn">模型有异议</span>{' '}
          <span className="small">{disagreement}</span>
        </div>
      )}

      {factors.length > 0 && (
        <div className="small muted" style={{ marginTop: 6 }}>
          并存因素：{factors.join('、')}
        </div>
      )}
    </div>
  )
}

function Conclusions({ attributionId, evalId }: { attributionId: number; evalId: number }) {
  const { data, error, loading } = useAsync(() => api.attribution(attributionId), [attributionId])
  const [openSample, setOpenSample] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const results = data?.results ?? []
  const shown = results.slice(offset, offset + PAGE)
  const nav = usePagedRecordNavigation({
    items: shown,
    total: results.length,
    responseOffset: offset,
    offset,
    limit: PAGE,
    selectedKey: openSample,
    itemKey: (result) => result.sample_id,
    onSelect: (result) => setOpenSample(result.sample_id),
    onOffsetChange: setOffset,
  })

  if (loading) return <Loading what="归因结论" />
  if (error) return <Failed error={error} />
  if (!data) return null

  // 只切显示不切数据，根因计数与平均延迟都在全集上算。
  // 样本链路覆盖整个结论框，带返回按钮。
  if (openSample) {
    return (
      <div className="panel" style={{ marginTop: 14 }}>
        <div className="spread">
          <h3 style={{ margin: 0 }}>样本 {openSample} 的链路</h3>
          <RecordNav
            hasPrevious={nav.hasPrevious}
            hasNext={nav.hasNext}
            onPrevious={nav.previous}
            onNext={nav.next}
            onBack={() => setOpenSample(null)}
            backLabel="返回结论列表"
            position={nav.position}
            total={data.results.length}
            busy={nav.navigating}
          />
        </div>
        <SampleChain
          key={openSample}
          evalId={evalId}
          sampleId={openSample}
        />
      </div>
    )
  }

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="row tight" style={{ justifyContent: 'flex-end' }}>
        <Timing
          startedAt={data.created_at}
          finishedAt={data.finished_at}
          latencyMs={data.latency_mean}
          perLabel="条"
        />
      </div>

      <div className="row tight" style={{ marginBottom: 10 }}>
        {Object.entries(data.count_by_root_cause).map(([cause, count]) => (
          <span key={cause}>
            <CauseTag cause={cause} /> <span className="small muted">{count}</span>
          </span>
        ))}
      </div>

      {(data.count_by_root_cause.not_a_failure ?? 0) > 0 && (
        <div className="note small">
          全量样本中有 {data.count_by_root_cause.not_a_failure} 条答案是对的；需要分析的异常是{' '}
          {data.results.length - (data.count_by_root_cause.not_a_failure ?? 0)} 条。
        </div>
      )}

      {(data.count_by_root_cause.generation_fallback ?? 0) > 0 && (
        <div className="note warn small">
          有 {data.count_by_root_cause.generation_fallback} 条是生成端拒答。
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>sample_id</th>
            <th>根因</th>
            <th>处置</th>
            <th>来源</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {shown.map((result) => (
            <tr key={result.sample_id} className={openSample === result.sample_id ? 'selected' : ''}>
              <td className="mono small">{result.sample_id}</td>
              <td>
                <CauseTag cause={result.root_cause} />
              </td>
              <td className="small muted">{result.remedy}</td>
              <td>
                <span className="tag">{result.rule_based ? '规则' : '规则+模型'}</span>
                {/* 模型对规则分类有异议时在表里就标出来。 */}
                {Boolean((result.evidence?.model as { disagreement?: unknown })?.disagreement) && (
                  <span className="tag warn">有异议</span>
                )}
                {Boolean(result.evidence?.model_error) && (
                  <span className="tag bad">模型失败</span>
                )}
              </td>
              <td>
                <button className="action small" onClick={() => setOpenSample(result.sample_id)}>
                  看链路
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Pager
        total={data.results.length}
        offset={offset}
        limit={PAGE}
        onChange={(next) => {
          setOffset(next)
          // 翻页时收起展开的样本，它不在新页上。
          setOpenSample(null)
        }}
      />

      {shown
        .filter((result) => result.narrative || result.evidence?.model || result.evidence?.model_error)
        .map((result) => <ModelVerdict key={result.sample_id} result={result} />)}
    </div>
  )
}

/** 一条样本的完整链路：证据、指标、响应、每篇 gold 的编译 diff。 */
function SampleChain({
  evalId,
  sampleId,
}: {
  evalId: number
  sampleId: string
}) {
  const { data, error, loading } = useAsync(
    () => api.evalSample(evalId, sampleId),
    [evalId, sampleId],
  )

  if (loading) return <Loading what="样本链路" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const question = String(data.detail.question ?? '')
  const goldPages = Object.entries(data.gold_pages)
  const goldDocuments = data.metric_interpretations.find(
    (item) => item.evidence?.gold_documents,
  )?.evidence?.gold_documents ?? []

  return (
    <div className="panel flat sample-chain" style={{ marginTop: 12 }}>
      <dl className="kv sample-overview">
        <dt>问题</dt>
        <dd>{question}</dd>
        <dt>参考答案</dt>
        <dd>{(data.detail.reference_answers as string[] | undefined)?.join(' / ') ?? '—'}</dd>
        <dt>系统答案</dt>
        <dd>{data.answer || '—'}</dd>
        <dt>回答模式</dt>
        <dd>
          <ModeTag mode={data.answer_mode} />
        </dd>
        <dt>Gold 文档</dt>
        <dd className="gold-documents-cell">
          <div className="gold-document-list">
            {goldDocuments.length === 0 && <span className="muted">无 Gold 文档标注</span>}
            {goldDocuments.map((document) => (
              <span key={document.doc_id} className="gold-document-chip">
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id}</span>
              </span>
            ))}
          </div>
        </dd>
      </dl>

      {data.judge_verdicts.length > 0 && (
        <div className="row tight" style={{ marginTop: 8 }}>
          {data.judge_verdicts.map((verdict) => (
            <span
              key={verdict.metric}
              className={`tag ${verdict.failure_kind ? 'bad' : verdict.score === null ? '' : 'ok'}`}
              title={verdict.failure_kind ?? JSON.stringify(verdict.detail)}
            >
              {verdict.metric} {verdict.failure_kind ?? verdict.score?.toFixed(2) ?? '无定义'}
            </span>
          ))}
        </div>
      )}

      <MetricInterpretations
        items={data.metric_interpretations}
        response={data.response ?? {}}
      />

      <div className="full-chain-section">
      <Collapsible title="完整链路">
        <Chain
          question={question}
          goldPages={goldPages}
          response={data.response ?? {}}
          goldDocIds={(data.detail.gold_doc_ids as string[] | undefined) ?? []}
        />
        <Collapsible title="完整响应">
          <pre className="block tall">{JSON.stringify(data.response, null, 2)}</pre>
        </Collapsible>
      </Collapsible>
      </div>

      {data.judge_verdicts.length > 0 && (
        <>
          <h4 style={{ marginTop: 14 }}>Judge 原始判据</h4>
          {data.judge_verdicts.map((verdict) => (
            <Collapsible key={verdict.metric} title={`${verdict.metric} 判据明细`}>
              <pre className="block">{JSON.stringify(verdict.detail, null, 2)}</pre>
            </Collapsible>
          ))}
        </>
      )}
    </div>
  )
}

function MetricInterpretations({
  items,
  response,
}: {
  items: EvalSampleDetail['metric_interpretations']
  response: Record<string, unknown>
}) {
  const [family, setFamily] = useState('全部')
  const families = ['全部', ...Array.from(new Set(items.map((item) => item.family_label)))]
  const visible = family === '全部' ? items : items.filter((item) => item.family_label === family)
  return (
    <section className="sample-metrics">
      <div className="spread sample-metrics-title">
        <h4>逐样本指标（{visible.length}/{items.length}）</h4>
        <label className="metric-family-select">
          <span className="small muted">分组</span>
          <select value={family} onChange={(event) => setFamily(event.target.value)}>
            {families.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
      </div>
      {items.length === 0 ? (
        <p className="small muted">这次评测没有选择指标。</p>
      ) : (
        <div className="metric-interpretation-grid">
          {visible.map((item) => renderMetric(item, response))}
        </div>
      )}
    </section>
  )
}

type MetricItem = EvalSampleDetail['metric_interpretations'][number]
type MetricProps = { item: MetricItem; response: Record<string, unknown> }

function renderMetric(item: MetricItem, response: Record<string, unknown>) {
  const base = item.name.split('@', 1)[0] ?? item.name
  const props = { key: item.name, item, response }
  switch (base) {
    case 'recall': return <RecallMetric {...props} />
    case 'ndcg': return <NdcgMetric {...props} />
    case 'hit': return <HitMetric {...props} />
    case 'full_coverage': return <FullCoverageMetric {...props} />
    case 'mrr': return <MrrMetric {...props} />
    case 'em': return <ExactMatchMetric {...props} />
    case 'f1': return <TokenF1Metric {...props} />
    case 'citation_precision': return <CitationPrecisionMetric {...props} />
    case 'citation_recall': return <CitationRecallMetric {...props} />
    case 'truncation_loss': return <TruncationLossMetric {...props} />
    case 'truncated_gold': return <TruncatedGoldMetric {...props} />
    case 'graph_neighbor_precision': return <GraphNeighborPrecisionMetric {...props} />
    case 'graph_exclusive_gold_share': return <GraphExclusiveGoldShareMetric {...props} />
    case 'graph_neighbor_gold_snippets': return <GraphNeighborGoldSnippetsMetric {...props} />
    case 'faithfulness': return <FaithfulnessMetric {...props} />
    case 'answer_relevancy': return <AnswerRelevancyMetric {...props} />
    case 'context_relevancy': return <ContextRelevancyMetric {...props} />
    case 'answer_correctness': return <AnswerCorrectnessMetric {...props} />
    default: return <UnknownMetric {...props} />
  }
}

function MetricHeading({ item }: { item: MetricItem }) {
  return (
    <div className="row tight metric-heading">
      <strong className="mono">{item.name}</strong>
      <span className="small muted">{item.family_label}</span>
      <strong className="mono metric-value metric-score">
        得分 {item.value === null ? '未计算' : item.value.toFixed(4)}
      </strong>
    </div>
  )
}

function RecallMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  const snippets = item.evidence?.snippets ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档片段（{retrieved.length}）</h5>
          {retrieved.length === 0 && <p className="small muted">没有检索到文档。</p>}
          {retrieved.map((document) => {
            const documentSnippets = snippets.filter((snippet) =>
              snippet.page_ids.includes(document.page_id),
            )
            return (
              <div key={`${document.rank}-${document.page_id}`} className="metric-evidence-document">
                <div>
                  <span className="mono small muted">#{document.rank}</span>{' '}
                  <strong>{document.title}</strong>{' '}
                  <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                  {document.is_gold && <span className="tag ok">gold</span>}
                  {!document.mapped && <span className="tag warn">未映射</span>}
                </div>
                {documentSnippets.length === 0 ? (
                  <p className="small muted">没有对应的 snippet 正文。</p>
                ) : documentSnippets.map((snippet) => (
                  <blockquote key={snippet.id ?? snippet.rank}>
                    <div className="readable-text">{snippet.text || '（无正文）'}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function NdcgMetric({ item }: MetricProps) {
  const documents = item.evidence?.documents ?? []
  const gains = new Map((item.evidence?.contributions ?? []).map((row) => [row.rank, row.gain]))
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details"><summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>
          <h5>参与排名计算的文档（{documents.length}）</h5>
          <div className="metric-document-list">
            {documents.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {documents.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}{' '}
                <span className="small muted">排名贡献 {(gains.get(document.rank) ?? 0).toFixed(4)}</span>
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function HitMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档（{retrieved.length}）</h5>
          <div className="metric-document-list">
            {retrieved.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {retrieved.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}
                {!document.mapped && <span className="tag warn">未映射</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function FullCoverageMetric({ item }: MetricProps) {
  const retrieved = item.evidence?.documents ?? []
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>参与计算的检索文档（{retrieved.length}）</h5>
          <div className="metric-document-list">
            {retrieved.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {retrieved.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}
                {!document.mapped && <span className="tag warn">未映射</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function MrrMetric({ item }: MetricProps) {
  const documents = item.evidence?.documents ?? []
  const firstGoldRank = documents.find((document) => document.is_gold)?.rank
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details"><summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>
          <h5>检索文档排名（{documents.length}）</h5>
          <div className="metric-document-list">
            {documents.length === 0 && <span className="small muted">没有检索到文档。</span>}
            {documents.map((document) => (
              <span key={`${document.rank}-${document.page_id}`} className="metric-document-item">
                <span className="mono small muted">#{document.rank}</span>{' '}
                <strong>{document.title}</strong>{' '}
                <span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}
                {document.is_gold && <span className="tag ok">gold</span>}{' '}
                {document.rank === firstGoldRank && <span className="tag accent">首个 Gold</span>}
              </span>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function ExactMatchMetric({ item, response }: MetricProps) {
  const references = item.evidence?.answer_comparison?.references ?? []
  const referenceTexts = references.map((reference) =>
    typeof reference === 'string' ? reference : reference.text,
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>系统答案</h5>
          <div className="metric-document-item readable-text">
            {String(response.answer ?? '') || '（空）'}
          </div>
          <h5>参考答案（{referenceTexts.length}）</h5>
          <div className="metric-document-list">
            {referenceTexts.length === 0 && <span className="small muted">无标答</span>}
            {referenceTexts.map((reference, index) => (
              <div key={index} className="metric-document-item readable-text">{reference}</div>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function TokenF1Metric({ item, response }: MetricProps) {
  const references = item.evidence?.answer_comparison?.references ?? []
  const referenceTexts = references.map((reference) =>
    typeof reference === 'string' ? reference : reference.text,
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>系统答案</h5>
          <div className="metric-document-item readable-text">
            {String(response.answer ?? '') || '（空）'}
          </div>
          <h5>参考答案（{referenceTexts.length}）</h5>
          <div className="metric-document-list">
            {referenceTexts.length === 0 && <span className="small muted">无参考答案</span>}
            {referenceTexts.map((reference, index) => (
              <div key={index} className="metric-document-item readable-text">{reference}</div>
            ))}
          </div>
        </div>
      </details>
    </article>
  )
}
function CitationPrecisionMetric({ item }: MetricProps) {
  const citations = item.evidence?.documents ?? []
  const excerptsByPage = new Map(
    (item.evidence?.citation_excerpts ?? []).map((citation) => [
      citation.page_id,
      citation.excerpts,
    ]),
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>Citation 文档片段（{citations.length}）</h5>
          {citations.length === 0 && <p className="small muted">没有 citation 文档片段。</p>}
          {citations.map((citation, index) => {
            const excerpts = excerptsByPage.get(citation.page_id) ?? []
            return (
              <div key={`${citation.page_id}-${index}`} className="metric-evidence-document">
                <div>
                  <strong>{citation.title}</strong>{' '}
                  <span className="mono small muted">{citation.doc_id ?? citation.page_id}</span>{' '}
                  {citation.is_gold && <span className="tag ok">gold</span>}
                </div>
                {excerpts.length === 0 ? (
                  <p className="small muted">没有证据片段。</p>
                ) : excerpts.map((text, excerptIndex) => (
                  <blockquote key={excerptIndex}>
                    <div className="readable-text">{text}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function CitationRecallMetric({ item }: MetricProps) {
  const citations = item.evidence?.documents ?? []
  const excerptsByPage = new Map(
    (item.evidence?.citation_excerpts ?? []).map((citation) => [
      citation.page_id,
      citation.excerpts,
    ]),
  )
  return (
    <article className={`metric-interpretation-card ${item.status}`}>
      <MetricHeading item={item} />
      <details className="metric-details">
        <summary className="action small">查看</summary>
        <div className="metric-details-body">
          <dl className="kv compact-kv">
            <dt>指标计算方式</dt>
            <dd>{item.evidence?.formula ?? item.description}</dd>
          </dl>
          <h5>实际引用的所有文档（{citations.length}）</h5>
          {citations.length === 0 && <p className="small muted">没有引用文档。</p>}
          {citations.map((citation) => {
            const excerpts = excerptsByPage.get(citation.page_id) ?? []
            return (
              <div key={`${citation.rank}-${citation.page_id}`} className="metric-evidence-document">
                <div>
                  <strong>{citation.title}</strong>{' '}
                  <span className="mono small muted">{citation.doc_id ?? citation.page_id}</span>{' '}
                  {citation.is_gold && <span className="tag ok">gold</span>}
                  {!citation.mapped && <span className="tag warn">未映射</span>}
                </div>
                {excerpts.length === 0 ? (
                  <p className="small muted">没有 citationEvidence 片段。</p>
                ) : excerpts.map((text, excerptIndex) => (
                  <blockquote key={excerptIndex}>
                    <div className="readable-text">{text}</div>
                  </blockquote>
                ))}
              </div>
            )
          })}
        </div>
      </details>
    </article>
  )
}
function TruncationLossMetric({ item }: MetricProps) {
  return <DocumentMetric item={item} title="已检索但未进入引用的文档" documents={item.evidence?.difference_documents ?? []} />
}
function TruncatedGoldMetric({ item }: MetricProps) {
  return <DocumentMetric item={item} title="已检索但未进入引用的 Gold 文档" documents={item.evidence?.difference_documents ?? []} />
}
function DocumentMetric({ item, title, documents }: { item: MetricItem; title: string; documents: NonNullable<NonNullable<MetricItem['evidence']>['documents']> }) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><h5>{title}（{documents.length}）</h5><div className="metric-document-list">{documents.length === 0 && <span className="small muted">没有文档。</span>}{documents.map((document) => <span key={`${document.rank}-${document.page_id}`} className="metric-document-item"><strong>{document.title}</strong>{' '}<span className="mono small muted">{document.doc_id ?? document.page_id}</span>{' '}{document.is_gold && <span className="tag ok">gold</span>}</span>)}</div></div></details></article>
}
function GraphExclusiveGoldShareMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
function GraphNeighborPrecisionMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
function GraphNeighborGoldSnippetsMetric({ item }: MetricProps) {
  const documents = graphHitDocuments(item)
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><GraphDocumentList documents={documents} /></div></details></article>
}
type GraphDocument = { docId: string; title: string; isGold: boolean; rank: number }

function graphHitDocuments(item: MetricItem): GraphDocument[] {
  const documents = new Map<string, GraphDocument>()
  for (const snippet of item.evidence?.snippets ?? []) {
    if (!snippet.is_graph) continue
    for (const docId of snippet.doc_ids) {
      const previous = documents.get(docId)
      documents.set(docId, {
        docId,
        title: previous?.title || snippet.title || docId,
        isGold: previous?.isGold || snippet.gold_doc_ids.includes(docId),
        rank: Math.min(previous?.rank ?? snippet.rank, snippet.rank),
      })
    }
  }
  return [...documents.values()].sort((left, right) => left.rank - right.rank)
}

function GraphDocumentList({ documents }: { documents: GraphDocument[] }) {
  return (
    <>
      <h5>图扩展命中的文档（{documents.length}）</h5>
      <div className="metric-document-list">
        {documents.length === 0 && <span className="small muted">没有图扩展命中的文档。</span>}
        {documents.map((document) => (
          <span key={document.docId} className="metric-document-item">
            <strong>{document.title}</strong>{' '}
            <span className="mono small muted">{document.docId}</span>{' '}
            {document.isGold && <span className="tag ok">gold</span>}
          </span>
        ))}
      </div>
    </>
  )
}
function FaithfulnessMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="上下文片段" mode="claims" snippets={item.evidence?.snippets ?? []} />
}
function AnswerRelevancyMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="答案句子" mode="sentences" />
}
function ContextRelevancyMetric({ item }: MetricProps) {
  return <JudgeMetric item={item} label="上下文段落" mode="passages" snippets={item.evidence?.snippets ?? []} />
}
function AnswerCorrectnessMetric({ item, response }: MetricProps) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl><h5>系统答案</h5><div className="metric-document-item readable-text">{String(response.answer ?? '') || '（空）'}</div><ReferenceAnswers item={item} /><JudgeVerdict item={item} /></div></details></article>
}
function UnknownMetric({ item }: MetricProps) {
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl></div></details></article>
}

function ReferenceAnswers({ item }: { item: MetricItem }) {
  const references = item.evidence?.answer_comparison?.references ?? []
  return <><h5>参考答案（{references.length}）</h5><div className="metric-document-list">{references.length === 0 && <span className="small muted">无参考答案</span>}{references.map((reference, index) => <div key={index} className="metric-document-item readable-text">{typeof reference === 'string' ? reference : reference.text}</div>)}</div></>
}

function JudgeMetric({ item, label, mode, snippets = [] }: { item: MetricItem; label: string; mode: 'claims' | 'sentences' | 'passages'; snippets?: NonNullable<NonNullable<MetricItem['evidence']>['snippets']> }) {
  const detail = item.evidence?.judge_detail ?? {}
  const entries = Array.isArray(detail[mode]) ? detail[mode] as Record<string, unknown>[] : []
  return <article className={`metric-interpretation-card ${item.status}`}><MetricHeading item={item} /><details className="metric-details"><summary className="action small">查看</summary><div className="metric-details-body"><dl className="kv compact-kv"><dt>指标计算方式</dt><dd>{item.evidence?.formula ?? item.description}</dd></dl>{snippets.length > 0 && <><h5>{label}（{snippets.length}）</h5>{snippets.map((snippet) => <div key={snippet.id ?? snippet.rank} className="metric-evidence-document"><strong>{snippet.title || `片段 #${snippet.rank}`}</strong><div className="readable-text">{snippet.text || '（无正文）'}</div></div>)}</>}<h5>Judge 明细（{entries.length}）</h5>{entries.length === 0 && <p className="small muted">无判定明细。</p>}{entries.map((entry, index) => <div key={index} className="metric-document-item"><div className="readable-text">{String(entry.claim ?? entry.sentence ?? `段落 ${String(entry.index ?? index + 1)}`)}</div><span className="tag">{String(entry.verdict ?? '')}</span></div>)}</div></details></article>
}

function JudgeVerdict({ item }: { item: MetricItem }) {
  const detail = item.evidence?.judge_detail ?? {}
  return <><h5>Judge 判定</h5><div className="metric-document-item"><strong>{String(detail.verdict ?? '无')}</strong>{detail.reason ? `：${String(detail.reason)}` : ''}</div></>
}

/** 检索信号的中文名。graph-neighbor 用强调色，它是图边的净贡献。 */
const REASONS: Record<string, { text: string; kind: string }> = {
  semantic: { text: '语义', kind: '' },
  lexical: { text: '词面', kind: '' },
  'exact-title': { text: '标题精确', kind: '' },
  'graph-neighbor': { text: '图扩展', kind: 'accent' },
  'sidecar-prefiltered': { text: '预筛', kind: '' },
}

/** 完整链路：原文 → artifact → 检索到的 chunk → 回答，一屏走完。 */
function Chain({
  question,
  goldPages,
  response,
  goldDocIds,
}: {
  question: string
  goldPages: [string, string | null][]
  response: Record<string, unknown>
  goldDocIds: string[]
}) {
  const snippets = (response.snippets as Snippet[] | undefined) ?? []
  const retrieved = (response.retrievedSources as { title?: string; id?: string }[] | undefined) ?? []
  const citations = (response.citations as { sourcePageId?: string }[] | undefined) ?? []
  const goldPageIds = new Set(goldPages.map(([, page]) => page).filter(Boolean))
  const cited = new Set(citations.map((c) => c.sourcePageId).filter(Boolean))

  const isGold = (snippet: Snippet) =>
    (snippet.sourceWindows ?? []).some((w) => w.sourcePageId && goldPageIds.has(w.sourcePageId))
  const graphOnly = snippets.filter((s) => (s.retrievalReasons ?? []).includes('graph-neighbor'))

  return (
    <>
      <div className="chain-summary small">
        <span>
          gold <strong>{goldDocIds.length}</strong> 篇
        </span>
        <span>→</span>
        <span>
          已编译 <strong>{goldPageIds.size}</strong> 篇
        </span>
        <span>→</span>
        <span>
          检索 <strong>{snippets.length || retrieved.length}</strong> 条
          {snippets.length > 0 && (
            <span className="muted">（命中 gold {snippets.filter(isGold).length}）</span>
          )}
        </span>
        <span>→</span>
        <span>
          引用 <strong>{citations.length}</strong> 条
        </span>
        {graphOnly.length > 0 && (
          <span className="muted">
            其中图扩展 <strong>{graphOnly.length}</strong> 条
          </span>
        )}
      </div>

      <h5>1 · 原文与编译产物</h5>
      {goldPages.length === 0 && <p className="small muted">这个数据集没有 gold 标注。</p>}
      {goldPages.map(([docId, pageId]) => (
        <Collapsible
          key={docId}
          title={
            <>
              {docId}
              {!pageId && <span className="tag bad">未导入</span>}
              {pageId && cited.has(pageId) && <span className="tag ok">被引用</span>}
            </>
          }
        >
          {pageId ? (
            <LineageView pageId={pageId} question={question} />
          ) : (
            <p className="small muted">这篇没有 page_id，编译链路看不了。</p>
          )}
        </Collapsible>
      ))}

      <h5>2 · 检索到的 chunk（{snippets.length}）</h5>
      {snippets.length === 0 && (
        <p className="small muted">
          响应里没有 snippets。
          {retrieved.length > 0 && `只有 ${retrieved.length} 条 retrievedSources（无正文）。`}
        </p>
      )}
      {snippets.map((snippet, index) => (
        <Collapsible
          key={index}
          title={
            <>
              <span className="mono small">#{index + 1}</span> {snippet.title || '（无标题）'}
              {isGold(snippet) && <span className="tag ok">gold</span>}
              {(snippet.retrievalReasons ?? []).map((reason) => (
                <span key={reason} className={`tag ${REASONS[reason]?.kind ?? ''}`}>
                  {REASONS[reason]?.text ?? reason}
                </span>
              ))}
            </>
          }
        >
          <pre className="block">{snippet.text || '（无正文）'}</pre>
        </Collapsible>
      ))}

    </>
  )
}
