import { useState } from 'react'
import { api } from '../api'
import type {
  AttributionResult,
  AttributionRun,
  CompileRun,
  EvalRun,
  MetricsView,
  Provider,
  Snippet,
} from '../types'
import {
  CauseTag,
  CleanupButton,
  Collapsible,
  Failed,
  Field,
  Loading,
  ModeTag,
  Pager,
  StatusTag,
  Timing,
  useAction,
  useAsync,
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
  onOpenSettings,
  onOpenTasks,
}: {
  activeEval: number | null
  onSelectEval: (id: number) => void
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
        <div className="panel">
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
        </div>
      )}

      {current && current.attributions.length > 0 && (
        <>
          <h3>归因记录</h3>
          <table className="records-table">
            <thead>
              <tr>
                <th>名称</th>
                <th>状态</th>
                <th>依据指标</th>
                <th className="num">样本数</th>
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
                  <td>
                    <StatusTag status={run.status} />
                  </td>
                  <td className="mono small">{run.metric}</td>
                  <td className="num">{run.sample_limit}</td>
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
  const metrics = useAsync<MetricsView>(() => api.metrics(), [])
  const providers = useAsync<Provider[]>(() => api.providers('attribution'), [])
  const [name, setName] = useState('')
  const [metric, setMetric] = useState('recall@5')
  const [limit, setLimit] = useState(10)
  const [useModel, setUseModel] = useState(false)
  const [providerId, setProviderId] = useState<number | ''>('')
  const start = useAction<unknown>()

  // 只列这一轮算过的指标，没算过的没有逐样本值可排。
  const candidates = (metrics.data?.definitions ?? [])
    .filter((d) => evalRun.metrics.includes(d.name))
    .flatMap((d) => (d.per_k ? evalRun.ks.map((k) => `${d.name}@${k}`) : [d.name]))

  return (
    <div style={{ marginTop: 12 }}>
      {start.error && <Failed error={start.error} />}
      <div className="row">
        <Field label="归因名称" hint="留空自动生成">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="自动" />
        </Field>
        <Field label="依据指标" hint="取该指标最差的 N 条">
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {candidates.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="样本数" hint="上限 50">
          <input
            type="number"
            min={1}
            max={50}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          />
        </Field>
        {useModel && (
          <Field label="归因模型">
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
          disabled={start.busy || candidates.length === 0}
          onClick={() =>
            start.run(async () => {
              const task = await api.startTask('attribute', {
                eval_id: evalRun.id,
                metric,
                sample_limit: limit,
                use_model: useModel,
                ...(name.trim() ? { name: name.trim() } : {}),
                ...(providerId === '' ? {} : { provider_id: providerId }),
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

// 逐样本表的每页行数。
const PAGE = 15

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

  if (loading) return <Loading what="归因结论" />
  if (error) return <Failed error={error} />
  if (!data) return null

  // 只切显示不切数据，根因计数与平均延迟都在全集上算。
  const shown = data.results.slice(offset, offset + PAGE)

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="panel-head">
        <h3>{data.name} 的根因</h3>
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
          有 {data.count_by_root_cause.not_a_failure} 条答案其实是对的 —— 它们进到最差 N
          只是因为检索指标低。真实失败是 {data.results.length -
            (data.count_by_root_cause.not_a_failure ?? 0)} 条。
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
                <button
                  className="action small"
                  onClick={() =>
                    setOpenSample(openSample === result.sample_id ? null : result.sample_id)
                  }
                >
                  {openSample === result.sample_id ? '收起' : '看链路'}
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

      {openSample && (
        <SampleChain
          key={openSample}
          evalId={evalId}
          sampleId={openSample}
          evidence={
            data.results.find((r) => r.sample_id === openSample)?.evidence ?? {}
          }
        />
      )}
    </div>
  )
}

/** 一条样本的完整链路：证据、指标、响应、每篇 gold 的编译 diff。 */
function SampleChain({
  evalId,
  sampleId,
  evidence,
}: {
  evalId: number
  sampleId: string
  evidence: Record<string, unknown>
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

  return (
    <div className="panel flat" style={{ marginTop: 12 }}>
      <dl className="kv">
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

      <Chain
        question={question}
        goldPages={goldPages}
        response={data.response ?? {}}
        goldDocIds={(data.detail.gold_doc_ids as string[] | undefined) ?? []}
      />

      <h4 style={{ marginTop: 14 }}>原始数据</h4>
      <Collapsible title="判据证据">
        <pre className="block">{JSON.stringify(evidence, null, 2)}</pre>
      </Collapsible>
      <Collapsible title="逐样本指标">
        <pre className="block">{JSON.stringify(data.metrics, null, 2)}</pre>
      </Collapsible>
      {data.judge_verdicts.map((verdict) => (
        <Collapsible key={verdict.metric} title={`${verdict.metric} 判据明细`}>
          <pre className="block">{JSON.stringify(verdict.detail, null, 2)}</pre>
        </Collapsible>
      ))}
      <Collapsible title="完整响应">
        <pre className="block tall">{JSON.stringify(data.response, null, 2)}</pre>
      </Collapsible>
    </div>
  )
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
      <h4 style={{ marginTop: 14 }}>完整链路</h4>
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
