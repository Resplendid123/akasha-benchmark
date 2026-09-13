import { useState } from 'react'
import { api } from '../api'
import type { BadcaseAnalysis, Provider, SampleLineage } from '../types'
import {
  CauseTag,
  Collapsible,
  Empty,
  Failed,
  Loading,
  ModeTag,
  metric,
  num,
  useAction,
  useAsync,
} from '../ui'
import { DiffPanels, LineageTables } from './DocDiff'

/** 归因层：完整链路 + 自动归因。
 *
 * 归因分两段。**规则在前**：判据全部来自已有指标与链路，不需要任何模型配置
 * 就能出结果。模型在后，补一段因果叙述。分开不是稳妥，是两者答的问题不同 ——
 * 规则给分类（哪一段断了），模型给叙述（为什么断在那里）。
 *
 * 顺序即优先级，而第一条是红线：generation_fallback 必须最先判。run001 上
 * 四条 recall@5 < 1.0 里三条是生成端拒答，判错顺序会把 1 条检索问题读成 4 条。
 */
export function Badcase({
  activeEval,
  onSelectEval,
  onOpenSettings,
}: {
  activeEval: number | null
  onSelectEval: (id: number) => void
  onOpenSettings: () => void
}) {
  const layers = useAsync(() => api.layers(), [])
  const evalLayers = (layers.data?.index_layers ?? []).flatMap((index) =>
    index.query_layers.flatMap((query) => query.eval_layers),
  )
  const current = activeEval ?? evalLayers[0]?.id ?? null

  if (layers.loading) return <Loading what="评测层" />
  if (layers.error) return <Failed error={layers.error} />

  return (
    <>
      <h2>归因层</h2>

      {evalLayers.length === 0 ? (
        <Empty>暂无评测批次。</Empty>
      ) : (
        <>
          <div className="row tight" style={{ marginBottom: 14 }}>
            {evalLayers.map((entry) => (
              <button
                key={entry.id}
                className={`action${current === entry.id ? ' primary' : ''}`}
                onClick={() => onSelectEval(entry.id)}
              >
                #{entry.id} {entry.label}
              </button>
            ))}
          </div>
          {current !== null && (
            <BadcaseBody evalLayerId={current} onOpenSettings={onOpenSettings} />
          )}
        </>
      )}
    </>
  )
}

function BadcaseBody({
  evalLayerId,
  onOpenSettings,
}: {
  evalLayerId: number
  onOpenSettings: () => void
}) {
  const existing = useAsync(() => api.badcases(evalLayerId), [evalLayerId])
  const analysis = useAsync(() => api.providers('analysis'), [])
  const [openSample, setOpenSample] = useState<string | null>(null)

  if (existing.loading) return <Loading what="归因结果" />
  if (existing.error) return <Failed error={existing.error} />
  if (!existing.data) return null

  const modelReady = (analysis.data ?? []).some((p: Provider) => p.api_key_set)
  const total = Object.values(existing.data.count_by_root_cause).reduce((a, b) => a + b, 0)

  return (
    <>
      <AnalysisPanel
        evalLayerId={evalLayerId}
        modelReady={modelReady}
        onOpenSettings={onOpenSettings}
        onDone={existing.reload}
      />

      {total === 0 ? (
        <Empty>暂无归因结果。</Empty>
      ) : (
        <>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>根因分布（{total} 条）</h3>
            <table>
              <thead>
                <tr>
                  <th>根因</th>
                  <th className="num">条数</th>
                  <th>处置</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(existing.data.count_by_root_cause).map(([cause, count]) => (
                  <tr key={cause}>
                    <td>
                      <CauseTag cause={cause} />
                    </td>
                    <td className="num">{count}</td>
                    <td className="small muted">{existing.data!.causes[cause]?.remedy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>逐条结论</h3>
          {existing.data.analyses.map((entry) => (
            <AnalysisCard
              key={entry.sample_id}
              entry={entry}
              open={openSample === entry.sample_id}
              onToggle={() =>
                setOpenSample(openSample === entry.sample_id ? null : entry.sample_id)
              }
              evalLayerId={evalLayerId}
            />
          ))}
        </>
      )}
    </>
  )
}

/** 跑归因：选指标、条数、要不要用模型。 */
function AnalysisPanel({
  evalLayerId,
  modelReady,
  onOpenSettings,
  onDone,
}: {
  evalLayerId: number
  modelReady: boolean
  onOpenSettings: () => void
  onDone: () => void
}) {
  const [metricName, setMetricName] = useState('recall@5')
  const [limit, setLimit] = useState(10)
  const [useModel, setUseModel] = useState(false)
  const action = useAction<{ analyzed: number; count_by_root_cause: Record<string, number> }>()

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>跑一批归因</h3>
      <div className="row">
        <label className="field">
          按哪个指标取最差
          <input value={metricName} onChange={(event) => setMetricName(event.target.value)} />
        </label>
        <label className="field">
          条数（上限 50）
          <input
            type="number"
            min={1}
            max={50}
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
          />
        </label>
        <label className="check" style={{ marginTop: 14 }}>
          <input
            type="checkbox"
            checked={useModel}
            disabled={!modelReady}
            onChange={(event) => setUseModel(event.target.checked)}
          />
          用分析模型补因果叙述
          {!modelReady && <span className="tag warn">未配置</span>}
        </label>
        <button
          className="action primary"
          style={{ marginTop: 12 }}
          disabled={action.busy}
          onClick={() =>
            action.run(async () => {
              const result = await api.analyzeWorst(
                evalLayerId,
                { metric: metricName, limit },
                useModel,
              )
              onDone()
              return result
            })
          }
        >
          {action.busy ? '归因中…' : '开始归因'}
        </button>
      </div>

      {!modelReady && (
        <div className="note plain small">
          没配分析模型也能跑 —— 规则归因的判据全部来自已有指标与链路。配了模型会在
          分类之上补一段因果叙述。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenSettings}>
            去配置
          </button>
        </div>
      )}
      {useModel && (
        <div className="note warn small">
          每条一次 LLM 调用，会花钱。上限压到 50 条是故意的摩擦。
        </div>
      )}
      {action.error && <div className="note bad">{action.error}</div>}
      {action.result && (
        <div className="note">
          归因了 {action.result.analyzed} 条：
          {Object.entries(action.result.count_by_root_cause)
            .map(([cause, count]) => `${cause}=${count}`)
            .join('、')}
        </div>
      )}
    </div>
  )
}

function AnalysisCard({
  entry,
  open,
  onToggle,
  evalLayerId,
}: {
  entry: BadcaseAnalysis
  open: boolean
  onToggle: () => void
  evalLayerId: number
}) {
  const evidence = entry.evidence

  return (
    <div className="panel">
      <div className="spread">
        <div>
          <CauseTag cause={entry.root_cause} />{' '}
          <span className="small mono muted">{entry.sample_id}</span>{' '}
          {entry.rule_based ? (
            <span className="tag" title="纯规则判定，未调模型">
              规则
            </span>
          ) : (
            <span className="tag accent" title="模型补了因果叙述">
              模型
            </span>
          )}
        </div>
        <button className="action small" onClick={onToggle}>
          {open ? '收起链路' : '看完整链路'}
        </button>
      </div>

      <div className="row small muted" style={{ marginTop: 6 }}>
        <span>
          <ModeTag mode={evidence.answer_mode} />
        </span>
        <span>hit {num(evidence.hit, 2)}</span>
        <span>coverage {num(evidence.full_coverage, 2)}</span>
        <span>
          gold {evidence.gold_count} / 召回 {evidence.retrieved_count} / 引用{' '}
          {evidence.citation_count}
        </span>
        <span>F1 {num(evidence.f1, 3)}</span>
        {evidence.truncated_gold > 0 && (
          <span className="tag warn">截断 gold {evidence.truncated_gold}</span>
        )}
      </div>

      {evidence.question_terms_lost.length > 0 && (
        <div className="note bad">
          <strong>编译产物里缺少问题中的实词：</strong>
          <div style={{ marginTop: 4 }}>
            {evidence.question_terms_lost.map((term) => (
              <span key={term} className="chip lost">
                {term}
              </span>
            ))}
          </div>
        </div>
      )}

      {!evidence.lineage_available && (
        <div className="note warn small">
          链路不可用（没配只读数据库），所以「编译丢词」这条判据没能参与判定。
          到「配置」层填 database_url 能让归因更准。
        </div>
      )}

      {entry.narrative && (
        <div className="note">
          {entry.narrative}
          {evidence.model?.disagreement && (
            <div className="small" style={{ marginTop: 6 }}>
              <strong>模型对规则分类有异议：</strong>
              {evidence.model.disagreement}
            </div>
          )}
          {evidence.model?.contributing_factors && (
            <div className="small muted" style={{ marginTop: 4 }}>
              相关因素：{evidence.model.contributing_factors.join('、')}
            </div>
          )}
        </div>
      )}

      {evidence.model_error && (
        <div className="note warn small">
          分析模型这一条没跑成：{evidence.model_error}。规则结论仍然有效。
        </div>
      )}

      {entry.remedy && <p className="small muted">{entry.remedy}</p>}

      {open && <FullChain evalLayerId={evalLayerId} sampleId={entry.sample_id} />}
    </div>
  )
}

/** 完整链路：样本 -> 响应 -> 每篇 gold 的 page -> artifact -> chunk -> 图边 -> 原文。 */
function FullChain({
  evalLayerId,
  sampleId,
}: {
  evalLayerId: number
  sampleId: string
}) {
  const detail = useAsync(() => api.sample(evalLayerId, sampleId), [evalLayerId, sampleId])
  const lineage = useAsync(
    () => api.sampleLineage(evalLayerId, sampleId).catch(() => null),
    [evalLayerId, sampleId],
  )

  if (detail.loading) return <Loading what="链路" />
  if (detail.error) return <Failed error={detail.error} />
  if (!detail.data) return null

  const sample = detail.data

  return (
    <div style={{ marginTop: 12 }}>
      <h4>① 问题与答案</h4>
      <dl className="kv">
        <dt>问题</dt>
        <dd>{sample.question}</dd>
        <dt>参考答案</dt>
        <dd>{sample.reference_answers.join(' / ')}</dd>
        <dt>系统答案</dt>
        <dd>{sample.answer ?? <span className="muted">（空）</span>}</dd>
        <dt>指标</dt>
        <dd className="small mono">
          recall@10 {metric(sample.metrics, 'recall@10')} · F1 {metric(sample.metrics, 'f1')} ·
          hit@10 {metric(sample.metrics, 'hit@10')}
        </dd>
      </dl>

      <h4>② gold 文档与它们的 page</h4>
      <table>
        <thead>
          <tr>
            <th>doc_id</th>
            <th>page_id</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(sample.gold_pages).map(([docId, pageId]) => (
            <tr key={docId}>
              <td className="small mono">{docId}</td>
              <td className="small mono muted">
                {pageId ?? <span className="tag bad">不在 page_map 里</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>③ 原文 → 编译产物</h4>
      {lineage.loading && <Loading what="编译链路" />}
      {lineage.data ? (
        <GoldChain lineage={lineage.data} />
      ) : (
        <div className="note warn small">
          拿不到编译链路 —— 需要 Akasha 的只读数据库连接（database_url）。
          其余部分照常可看。
        </div>
      )}

      <Collapsible title="④ 完整响应体与审计记录">
        <pre className="block tall">{JSON.stringify(sample.response, null, 2)}</pre>
        {sample.audit && (
          <>
            <h4>审计记录（retrievalDiagnostics）</h4>
            <pre className="block">{JSON.stringify(sample.audit, null, 2)}</pre>
          </>
        )}
      </Collapsible>

      {sample.judge_verdicts.length > 0 && (
        <Collapsible title={`judge 判决（${sample.judge_verdicts.length}）`}>
          {sample.judge_verdicts.map((verdict) => (
            <div key={verdict.metric} style={{ marginBottom: 8 }}>
              <div className="small">
                <span className="tag">{verdict.metric}</span>{' '}
                {verdict.score === null ? (
                  <span className="tag bad">{verdict.failure_kind ?? '无分数'}</span>
                ) : (
                  <strong className="mono">{verdict.score.toFixed(3)}</strong>
                )}
              </div>
              {verdict.reasoning?.claims && (
                <table>
                  <thead>
                    <tr>
                      <th>陈述</th>
                      <th>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {verdict.reasoning.claims.map((claim, index) => (
                      <tr key={index}>
                        <td className="small">{claim.claim}</td>
                        <td className="small">
                          <span
                            className={`tag ${claim.verdict === 'supported' ? 'ok' : 'bad'}`}
                          >
                            {claim.verdict}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
        </Collapsible>
      )}
    </div>
  )
}

function GoldChain({ lineage }: { lineage: SampleLineage }) {
  return (
    <>
      {lineage.gold.map((entry) => (
        <div key={entry.doc_id} style={{ marginBottom: 14 }}>
          <div className="small">
            <span className="tag accent">{entry.doc_id}</span>{' '}
            {entry.error && <span className="tag bad">{entry.error}</span>}
          </div>
          {entry.diff && <DiffPanels diff={entry.diff} />}
          {entry.lineage && (
            <Collapsible
              title={`链路明细：${entry.lineage.counts.artifacts} artifact · ${entry.lineage.counts.chunks} chunk · ${entry.lineage.counts.edges} 图边`}
            >
              <LineageTables lineage={entry.lineage} />
            </Collapsible>
          )}
        </div>
      ))}
    </>
  )
}
