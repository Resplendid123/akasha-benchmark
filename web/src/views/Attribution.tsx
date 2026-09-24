import { useEffect, useState } from 'react'
import { api } from '../api'
import type {
  AttributionResult,
  AttributionRun,
  CompileRun,
  EvidenceChain,
  EvalRun,
  Provider,
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
  RecordSearch,
  StatusTag,
  Timing,
  useAction,
  useAsync,
  usePagedRecordNavigation,
  usePoll,
} from '../ui'
import { AttributionChain } from './AttributionChain'
import { MetricInterpretations } from './MetricInterpretations'

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
                <th>整体报告</th>
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
                  <td className="small">{run.model_label ?? '—'}</td>
                  <td>
                    <StatusTag status={run.status} />
                  </td>
                  <td className="num">{run.sample_count}</td>
                  <td className="num">{run.success_count}</td>
                  <td>
                    <span className={`tag ${run.report_provider_id !== null ? 'ok' : ''}`}>
                      {run.report_provider_id !== null ? '已请求' : '未使用模型'}
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
          </>
        )}
      </div>

      <label className="check" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={useModel} onChange={() => setUseModel(!useModel)} />
        用模型分析本次评测的全部指标并生成一份整体报告（每轮一次调用）
      </label>

      {useModel && (providers.data ?? []).length === 0 && (
        <div className="note warn">
          还没配归因模型端点；不配也能跑逐样本规则归因，只是没有整体分析报告。
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
const ROOT_CAUSE_ORDER = [
  'answer_correct',
  'answer_incorrect',
  'generation_ignored_retrieval',
  'generation_fallback',
  'retrieval_evidence_incomplete',
  'compiled_away',
  'citation_dropped',
  'retrieval_miss',
  'graph_edge_missing',
  'unknown',
]

function exportFilename(name: string): string {
  const cleaned = name.trim().replace(/[<>:"/\\|?*\x00-\x1f]/g, '-').replace(/\s+/g, '-')
  return cleaned || 'attribution'
}

function downloadText(content: string, filename: string, type: string): void {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function Conclusions({ attributionId, evalId }: { attributionId: number; evalId: number }) {
  const { data, error, loading } = useAsync(() => api.attribution(attributionId), [attributionId])
  const [openSample, setOpenSample] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const [rootCause, setRootCause] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const results = data?.results ?? []
  const filteredResults = results.filter((result) => {
    if (rootCause && result.root_cause !== rootCause) return false
    if (!search.trim()) return true
    const needle = search.trim().toLowerCase()
    return [result.sample_id, result.dataset, result.root_cause, JSON.stringify(result.evidence)]
      .some((value) => value.toLowerCase().includes(needle))
  })
  const shown = filteredResults.slice(offset, offset + PAGE)
  const nav = usePagedRecordNavigation({
    items: shown,
    total: filteredResults.length,
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

  const causeEntries = Object.entries(data.count_by_root_cause).sort(
    ([left], [right]) => ROOT_CAUSE_ORDER.indexOf(left) - ROOT_CAUSE_ORDER.indexOf(right),
  )

  // 根因筛选只影响样本列表和前后导航，计数始终来自归因全集。
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
            total={filteredResults.length}
            busy={nav.navigating}
          />
        </div>
        <SampleChain
          key={openSample}
          evalId={evalId}
          sampleId={openSample}
          attributionResult={results.find((result) => result.sample_id === openSample)}
        />
      </div>
    )
  }

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread" style={{ marginBottom: 10 }}>
        <div className="row tight">
          <button
            className="action small"
            disabled={!data.report}
            title={data.report ? '下载整体评测分析报告' : '本次归因没有模型报告'}
            onClick={() => {
              if (!data.report) return
              const title = `# ${data.name} · 整体评测分析报告`
              const metadata = `归因 ID：${data.id}  \n评测 ID：${data.eval_id}  \n生成时间：${data.finished_at ?? data.created_at}`
              downloadText(
                `${title}\n\n${metadata}\n\n${data.report}\n`,
                `${exportFilename(data.name)}-report.md`,
                'text/markdown;charset=utf-8',
              )
            }}
          >
            导出整体报告
          </button>
          <button
            className="action small"
            disabled={results.length === 0}
            title="下载本次运行的全部逐样本规则归因，不受当前分组筛选影响"
            onClick={() => {
              const exported = {
                export_type: 'rule_attribution',
                attribution: {
                  id: data.id,
                  name: data.name,
                  eval_id: data.eval_id,
                  created_at: data.created_at,
                  finished_at: data.finished_at,
                },
                count_by_root_cause: data.count_by_root_cause,
                results: results.map((result) => ({
                  sample_id: result.sample_id,
                  dataset: result.dataset,
                  root_cause: result.root_cause,
                  evidence: result.evidence,
                })),
              }
              downloadText(
                `${JSON.stringify(exported, null, 2)}\n`,
                `${exportFilename(data.name)}-rules.json`,
                'application/json;charset=utf-8',
              )
            }}
          >
            导出规则归因
          </button>
        </div>
        <Timing
          startedAt={data.created_at}
          finishedAt={data.finished_at}
          latencyMs={data.report_latency_ms}
          perLabel="份报告"
        />
      </div>

      {(data.report || data.report_error) && (
        <section className={`note ${data.report_error ? 'warn' : 'plain'}`} style={{ marginBottom: 14 }}>
          <h3 style={{ marginTop: 0 }}>整体评测分析报告</h3>
          {data.report_error && (
            <div className="small" style={{ color: 'var(--bad)' }}>
              报告生成失败：{data.report_error}。逐样本规则归因不受影响。
            </div>
          )}
          {data.report && <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{data.report}</div>}
        </section>
      )}

      <div className="row tight" style={{ marginBottom: 10 }}>
        <RecordSearch
          placeholder="搜索 sample_id、数据集、根因或证据"
          value={search}
          onChange={setSearch}
          onSearch={(value) => {
            setSearch(value)
            setOffset(0)
            setOpenSample(null)
          }}
        />
        <button
          className={`action small ${rootCause === null ? 'primary' : ''}`}
          onClick={() => {
            setRootCause(null)
            setOffset(0)
            setOpenSample(null)
          }}
        >
          全部 <span className="small">{results.length}</span>
        </button>
        {causeEntries.map(([cause, count]) => (
          <button
            key={cause}
            className={`action small ${rootCause === cause ? 'primary' : ''}`}
            onClick={() => {
              setRootCause(rootCause === cause ? null : cause)
              setOffset(0)
              setOpenSample(null)
            }}
          >
            <CauseTag cause={cause} plain /> <span className="small">{count}</span>
          </button>
        ))}
      </div>

      <table>
        <thead>
          <tr>
            <th>sample_id</th>
            <th>根因</th>
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
              <td>
                <span className="tag">规则</span>
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
        total={filteredResults.length}
        offset={offset}
        limit={PAGE}
        onChange={(next) => {
          setOffset(next)
          setOpenSample(null)
        }}
      />
    </div>
  )
}

function EvidenceChainPanel({ evidence }: { evidence: unknown }) {
  if (!evidence || typeof evidence !== 'object') return null
  const chain = evidence as EvidenceChain
  if (!chain.steps?.length) return null
  const statusKind = chain.status === 'complete' ? 'ok' : chain.status === 'incomplete' ? 'bad' : 'warn'
  return (
    <section className="evidence-chain-panel">
      <div className="spread"><h4 style={{ margin: 0 }}>逐跳证据诊断</h4><span className={`tag ${statusKind}`}>{chain.status}</span></div>
      <div className="small muted" style={{ marginTop: 5 }}>{chain.supported_step_count ?? 0} supported · {chain.partial_step_count ?? 0} partial · {chain.missing_step_count ?? 0} missing{chain.model_false_negative_candidate && ' · general 但证据链完整'}</div>
      <div className="evidence-chain-steps">
        {chain.steps.map((step) => {
          const kind = step.status === 'supported' ? 'ok' : step.status === 'missing' ? 'bad' : 'warn'
          return <div className="evidence-chain-step" key={step.position}>
            <div className="spread"><strong>Step {step.position}</strong><span className={`tag ${kind}`}>{step.status}</span></div>
            <div className="small" style={{ marginTop: 5 }}>{step.question || '（无子问题）'}</div>
            <dl className="kv compact-kv"><dt>答案</dt><dd>{step.answer || '—'}</dd><dt>支持文档</dt><dd>{step.support_title || '—'} {step.support_doc_id && <span className="mono small muted">{step.support_doc_id}</span>}</dd><dt>答案出现</dt><dd>{step.answer_present ? '是' : '否'}</dd><dt>support 覆盖</dt><dd>{typeof step.support_token_recall === 'number' ? `${Math.round(step.support_token_recall * 100)}%` : '—'}</dd><dt>检索到相关论断</dt><dd>{step.claim_retrieved ? '是' : '否'}</dd></dl>
            {step.retrieved_evidence?.length ? (
              <div className="evidence-chain-claims">
                <strong className="small">检索到的相关论断</strong>
                {step.retrieved_evidence.map((evidence, index) => (
                  <blockquote key={`${evidence.source_type}-${evidence.title}-${index}`}>
                    <div className="row tight">
                      <span className={`tag ${evidence.source_type === 'context' ? 'accent' : ''}`}>
                        {evidence.source_type === 'context' ? 'Context' : '原文窗口'}
                      </span>
                      <span className="small muted">{evidence.title || '检索片段'}</span>
                    </div>
                    <div className="readable-text">{evidence.text || '（仅命中文档标题）'}</div>
                  </blockquote>
                ))}
              </div>
            ) : <div className="small muted" style={{ marginTop: 6 }}>未检索到可展示的相关论断。</div>}
          </div>
        })}
      </div>
    </section>
  )
}

function SampleChain({
  evalId,
  sampleId,
  attributionResult,
}: {
  evalId: number
  sampleId: string
  attributionResult?: AttributionResult
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

      <EvidenceChainPanel evidence={attributionResult?.evidence?.evidence_chain} />

      <MetricInterpretations
        items={data.metric_interpretations}
        response={data.response ?? {}}
      />

      <div className="full-chain-section">
        <Collapsible title="完整链路">
          <AttributionChain
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
