import { useState } from 'react'
import { api } from '../api'
import type { DatasetEval, EvalLayerDetail, MetricDefinition, SampleRow } from '../types'
import {
  Bar,
  CauseTag,
  Empty,
  Failed,
  Loading,
  ModeTag,
  metric,
  percent,
  useAsync,
} from '../ui'

/** 报告层：本轮所选指标的结果，以及任意样本。
 *
 * 两条口径贯穿这一页：
 *
 * 1. **每张检索表都给两份** —— 全样本，以及只算 answerMode == knowledge 的切片。
 *    差值就是生成端拒答的规模，不是检索失败。
 * 2. **算不了的指标连同原因一起显示，不画成 0。** 缺依赖与没勾选是两种原因,
 *    要分开陈述。
 */
export function Report({
  activeEval,
  onSelectEval,
  onOpenBadcase,
}: {
  activeEval: number | null
  onSelectEval: (id: number) => void
  onOpenBadcase: (id: number) => void
}) {
  const layers = useAsync(() => api.layers(), [])

  const evalLayers = (layers.data?.index_layers ?? []).flatMap((index) =>
    index.query_layers.flatMap((query) =>
      query.eval_layers.map((entry) => ({ ...entry, indexLabel: index.label })),
    ),
  )
  const current = activeEval ?? evalLayers[0]?.id ?? null

  if (layers.loading) return <Loading what="评测层" />
  if (layers.error) return <Failed error={layers.error} />

  return (
    <>
      <h2>报告</h2>
      {evalLayers.length === 0 ? (
        <Empty>还没有评测层。到「评测层」勾好指标跑一轮。</Empty>
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
            <ReportBody evalLayerId={current} onOpenBadcase={onOpenBadcase} />
          )}
        </>
      )}
    </>
  )
}

function ReportBody({
  evalLayerId,
  onOpenBadcase,
}: {
  evalLayerId: number
  onOpenBadcase: (id: number) => void
}) {
  const { data, error, loading } = useAsync(() => api.evalLayer(evalLayerId), [evalLayerId])
  const definitions = useAsync(() => api.metricDefinitions(), [])
  const [openDataset, setOpenDataset] = useState<string | null>(null)

  if (loading || definitions.loading) return <Loading what="报告" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const totalBadcases = Object.values(data.badcase_causes).reduce((a, b) => a + b, 0)

  return (
    <>
      <div className="note plain">
        <strong>这些数字该怎么读。</strong> Akasha 的召回跑在<strong>编译产物</strong>上而不是
        原文。编译可能改写或遗漏信息；与公开 baseline 比较前需对齐语料、样本、
        检索单位和答案格式。原文基线可帮助隔离编译影响。
        <div className="small" style={{ marginTop: 6 }}>
          Exact Match 要求归一化后的整段答案与参考答案相等。解释性长答案通常得分较低，
          但并非必然为零；请结合 F1、引用证据和人工抽查解读。
        </div>
      </div>

      <div className="panel">
        <div className="spread">
          <div>
            <strong>#{data.id}</strong> <span className="tag accent">{data.label}</span>
          </div>
          <div className="row tight small muted">
            <span className="mono">config {data.config_hash}</span>
            <span>k = {data.ks.join(', ')}</span>
            <span>勾选 {data.metrics.length} 项指标</span>
            {totalBadcases > 0 && (
              <button className="action small" onClick={() => onOpenBadcase(data.id)}>
                已归因 {totalBadcases} 条
              </button>
            )}
          </div>
        </div>

        {totalBadcases > 0 && (
          <div className="row tight" style={{ marginTop: 8 }}>
            {Object.entries(data.badcase_causes).map(([cause, count]) => (
              <span key={cause}>
                <CauseTag cause={cause} /> <span className="small muted">{count}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {data.judge.total > 0 && <JudgePanel judge={data.judge} />}

      {data.datasets.map((entry) => (
        <DatasetReport
          key={entry.dataset}
          entry={entry}
          definitions={definitions.data ?? []}
          selected={data.metrics}
          ks={data.ks}
          open={openDataset === entry.dataset}
          onToggle={() =>
            setOpenDataset(openDataset === entry.dataset ? null : entry.dataset)
          }
          evalLayerId={data.id}
        />
      ))}
    </>
  )
}

function JudgePanel({ judge }: { judge: EvalLayerDetail['judge'] }) {
  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>{judge.metric}</h3>
        <span className="small muted">
          均值的分母是 scored，不是 total —— 失败的那些被排除而不是记 0
        </span>
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <span>
          均值 <strong className="mono">{judge.mean === null ? '—' : judge.mean.toFixed(4)}</strong>
        </span>
        <span className="small muted">已打分 {judge.scored}</span>
        <span className="small muted">排除 {judge.excluded}</span>
        <span className={`tag ${judge.failure_rate > 0.1 ? 'bad' : 'ok'}`}>
          失败率 {percent(judge.failure_rate)}
        </span>
      </div>
      {Object.keys(judge.failures_by_kind).length > 0 && (
        <p className="small muted">
          失败分类：
          {Object.entries(judge.failures_by_kind)
            .map(([kind, count]) => `${kind}=${count}`)
            .join('、')}
          。四类处置完全不同 —— rate_limit 与 timeout 退避重试，parse_error 要改 prompt，
          refusal 是内容触发了安全策略。记 0 会让限流伪装成质量差。
        </p>
      )}
    </div>
  )
}

function DatasetReport({
  entry,
  definitions,
  selected,
  ks,
  open,
  onToggle,
  evalLayerId,
}: {
  entry: DatasetEval
  definitions: MetricDefinition[]
  selected: string[]
  ks: number[]
  open: boolean
  onToggle: () => void
  evalLayerId: number
}) {
  const overall = entry.scopes.overall ?? {}
  const knowledge = entry.scopes.knowledge_only ?? {}

  // 只显示这一轮实际算出来的指标 —— 报告的列以勾选为准。
  const shown = expandNames(definitions, selected, ks).filter(
    (name) => overall[name] !== undefined,
  )

  return (
    <div className="panel">
      <div className="spread">
        <h3 style={{ margin: 0 }}>{entry.dataset}</h3>
        <div className="row tight small muted">
          <span>
            {entry.responses_evaluated}/{entry.samples_in_subset} 条
          </span>
          {entry.http_failures > 0 && (
            <span className="tag bad">{entry.http_failures} 次 HTTP 失败</span>
          )}
          <button className="action small" onClick={onToggle}>
            {open ? '收起样本' : '看样本'}
          </button>
        </div>
      </div>

      <div className="row tight small" style={{ marginTop: 6 }}>
        <span className="muted">回答模式：</span>
        {Object.entries(entry.answer_mode_distribution).map(([mode, count]) => (
          <span key={mode}>
            <ModeTag mode={mode} /> <span className="muted">{count}</span>
          </span>
        ))}
      </div>

      {/* 两种「没有值」分开显示：算不了是数据集的性质，没勾选是这一轮的选择。
          混成一句会让人以为其他组也缺 gold 标注。 */}
      {entry.omitted_metrics.length > 0 && (
        <div className="note warn">
          <strong>{entry.omitted_metrics.length} 项指标算不了，已省略而不是报 0。</strong>
          <div className="small mono muted" style={{ marginTop: 4 }}>
            {entry.omitted_metrics.join('、')}
          </div>
        </div>
      )}
      {entry.omission_reason && (
        <div className="note plain small">{entry.omission_reason}</div>
      )}

      {shown.length > 0 ? (
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>指标</th>
              <th className="num">全样本</th>
              <th className="num">仅 knowledge</th>
              <th>差值的含义</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((name) => {
              const definition = definitions.find((d) => d.name === name.split('@')[0])
              const both = overall[name] !== undefined && knowledge[name] !== undefined
              const gap = both ? (knowledge[name] ?? 0) - (overall[name] ?? 0) : null
              return (
                <tr key={name}>
                  <td className="small mono" title={definition?.description}>
                    {name}
                    {definition && !definition.higher_is_better && (
                      <span className="tag" style={{ marginLeft: 4 }} title="越低越好">
                        ↓
                      </span>
                    )}
                  </td>
                  <td className="num">{metric(overall, name)}</td>
                  <td className="num">{metric(knowledge, name)}</td>
                  <td className="small muted">
                    {gap !== null && Math.abs(gap) > 0.001 ? (
                      <>
                        +{gap.toFixed(4)} —— 这部分差距来自生成端拒答，不是检索失败
                      </>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      ) : (
        <p className="small muted">这个数据集这一轮没有可显示的指标。</p>
      )}

      {entry.stratified && <Strata stratified={entry.stratified} />}

      {entry.unmapped_page_ids.length > 0 && (
        <div className="note warn small">
          有 {entry.unmapped_page_ids.length} 个被检索到的 page id 不在 page_map 里。
          它们仍占据排名位次，但永远不可能被算作 gold。
        </div>
      )}

      {open && <Samples evalLayerId={evalLayerId} dataset={entry.dataset} />}
    </div>
  )
}

function Strata({ stratified }: { stratified: NonNullable<DatasetEval['stratified']> }) {
  const rows = Object.entries(stratified.strata)
  if (rows.length <= 1) return null
  return (
    <>
      <h4>按 {stratified.key} 分层</h4>
      <table>
        <thead>
          <tr>
            <th>分层</th>
            <th className="num">n</th>
            <th className="num">recall@10</th>
            <th className="num">full_coverage@10</th>
            <th className="num">F1</th>
            <th>knowledge 占比</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, bucket]) => (
            <tr key={name}>
              <td className="small">{name}</td>
              <td className="num">{bucket.count}</td>
              <td className="num">{metric(bucket.metrics, 'recall@10')}</td>
              <td className="num">{metric(bucket.metrics, 'full_coverage@10')}</td>
              <td className="num">{metric(bucket.metrics, 'f1')}</td>
              <td style={{ minWidth: 90 }}>
                <Bar value={bucket.knowledge_answer_share} />
                <span className="small muted">{percent(bucket.knowledge_answer_share)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small muted">
        full_coverage 比 recall 均值更贴近多跳的实际需求：多跳少一跳就答不对，
        而 recall 在 2 篇 gold 上只有 0/0.5/1 三个取值。
      </p>
    </>
  )
}

/** 样本列表。**按 answerMode 分组是默认行为**，不是筛选器。 */
function Samples({ evalLayerId, dataset }: { evalLayerId: number; dataset: string }) {
  const { data, error, loading } = useAsync(
    () => api.samples(evalLayerId, { dataset }),
    [evalLayerId, dataset],
  )

  if (loading) return <Loading what="样本" />
  if (error) return <Failed error={error} />
  if (!data) return null

  return (
    <div style={{ marginTop: 12 }}>
      <div className="note plain small">{data.note}</div>
      {Object.entries(data.by_answer_mode).map(([mode, rows]) => (
        <div key={mode} className="mode-group">
          <div className="mode-head">
            <ModeTag mode={mode === 'missing' ? null : mode} />
            <span className="small muted">{rows.length} 条</span>
            {mode !== 'knowledge' && (
              <span className="small muted">
                检索得分按定义为 0（retrievedSources 被无条件清空）
              </span>
            )}
          </div>
          <SampleTable rows={rows} />
        </div>
      ))}
    </div>
  )
}

function SampleTable({ rows }: { rows: SampleRow[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>sample_id</th>
          <th className="num">gold</th>
          <th className="num">召回</th>
          <th className="num">引用</th>
          <th className="num">recall@10</th>
          <th className="num">F1</th>
          <th>答案</th>
        </tr>
      </thead>
      <tbody>
        {rows.slice(0, 50).map((row) => (
          <tr key={row.sample_id}>
            <td className="small mono truncate">{row.sample_id}</td>
            <td className="num">{row.gold_count}</td>
            <td className="num">{row.retrieved_count}</td>
            <td className="num">{row.citation_count}</td>
            <td className="num">{metric(row.metrics, 'recall@10')}</td>
            <td className="num">{metric(row.metrics, 'f1')}</td>
            <td className="small muted truncate">{row.answer ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** 把勾选的模板名展开成实际指标名（recall -> recall@2、recall@5 ...）。 */
function expandNames(
  definitions: MetricDefinition[],
  selected: string[],
  ks: number[],
): string[] {
  const names: string[] = []
  for (const name of selected) {
    const definition = definitions.find((d) => d.name === name)
    if (definition?.per_k) names.push(...ks.map((k) => `${name}@${k}`))
    else names.push(name)
  }
  return names
}
