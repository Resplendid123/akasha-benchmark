import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { IndexLayer, MetricDefinition, Provider, QueryLayerSummary } from '../types'
import {
  DatasetPicker,
  Empty,
  Failed,
  Loading,
  ModeTag,
  Pager,
  useAction,
  useAsync,
} from '../ui'

/** 评测层：勾指标、选编译数据、配问答与 judge 模型、看生成结果。
 *
 * 指标勾选范围由**所选数据集的标注**决定，不由数据集名字决定：数据集声明
 * 拥有什么，指标声明需要什么，闸门做集合比对。所以勾了只对部分组成立的指标
 * 不会报错 —— 缺依赖的那组会省略它并写明原因，而不是伪造 0 分。
 */
export function Evaluate({
  activeLayer,
  onSelectLayer,
  onOpenReport,
  onOpenSettings,
  onOpenTasks,
}: {
  activeLayer: number | null
  onSelectLayer: (id: number) => void
  onOpenReport: (evalLayerId: number) => void
  onOpenSettings: () => void
  onOpenTasks: () => void
}) {
  const layers = useAsync(() => api.layers(), [])
  const definitions = useAsync(() => api.metricDefinitions(), [])

  if (layers.loading || definitions.loading) return <Loading what="评测层" />
  if (layers.error) return <Failed error={layers.error} />
  if (definitions.error) return <Failed error={definitions.error} />

  const all = layers.data?.index_layers ?? []
  const ready = all.filter((layer) => layer.ready_for_query)
  const layer = all.find((entry) => entry.id === activeLayer) ?? ready[0] ?? all[0]

  return (
    <>
      <h2>评测层</h2>
      <p className="lede">
        一轮评测 = 一批编译好的数据 + 一组勾选的指标 + 问答模型（+ judge 模型）。
        answer 模型改了不需要重编译，所以它在这一层而不是编译层。
      </p>

      <QuickCheck onOpenTasks={onOpenTasks} />

      {all.length === 0 ? (
        <Empty>还没有编译层。先到「编译层」建一层并导入。</Empty>
      ) : (
        <>
          <LayerSelector
            layers={all}
            active={layer?.id ?? null}
            onSelect={onSelectLayer}
          />
          {layer && (
            <RunPanel
              layer={layer}
              definitions={definitions.data ?? []}
              onOpenSettings={onOpenSettings}
              onDone={layers.reload}
            />
          )}
          {layer && (
            <QueryLayers layers={layer.query_layers} onOpenReport={onOpenReport} />
          )}
        </>
      )}
    </>
  )
}

function LayerSelector({
  layers,
  active,
  onSelect,
}: {
  layers: IndexLayer[]
  active: number | null
  onSelect: (id: number) => void
}) {
  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>选编译后的数据</h3>
      <div className="row tight">
        {layers.map((layer) => (
          <button
            key={layer.id}
            className={`action${active === layer.id ? ' primary' : ''}`}
            disabled={!layer.ready_for_query}
            title={
              layer.ready_for_query
                ? `${layer.label}（${Object.values(layer.page_map_counts).reduce((a, b) => a + b, 0)} 篇）`
                : layer.not_ready_reasons.join('；')
            }
            onClick={() => onSelect(layer.id)}
          >
            #{layer.id} {layer.label}
            {!layer.ready_for_query && ' ·未就绪'}
          </button>
        ))}
      </div>
      {layers.every((layer) => !layer.ready_for_query) && (
        <div className="note warn">
          没有可用的编译层。半成品索引会产出一份「recall 低、拒答率高」的报告，
          那看起来像配置差、实际是索引没建好 —— 所以这里不放行。
        </div>
      )}
    </div>
  )
}

/** 起一轮评测：勾指标、选数据集、配 judge。 */
function RunPanel({
  layer,
  definitions,
  onOpenSettings,
  onDone,
}: {
  layer: IndexLayer
  definitions: MetricDefinition[]
  onOpenSettings: () => void
  onDone: () => void
}) {
  const datasetNames = useMemo(() => layer.datasets.map((d) => d.dataset), [layer])
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(datasetNames)
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [ks, setKs] = useState('2,5,10')
  const [queryLabel, setQueryLabel] = useState(`${layer.label}-query`)
  const [evalLabel, setEvalLabel] = useState(`${layer.label}-query-eval`)

  const available = useAsync(
    () =>
      selectedDatasets.length > 0
        ? api.availableMetrics(selectedDatasets)
        : Promise.resolve(null),
    [selectedDatasets.join(',')],
  )
  const judge = useAsync(() => api.providers('judge'), [])
  const startQuery = useAction<{ id: number }>()
  const startEval = useAction<{ id: number }>()

  // 数据集换了，勾选范围跟着变 —— 把已经不可用的指标去掉。
  useEffect(() => {
    if (!available.data) return
    const usable = new Set(available.data.computable_for_some)
    setSelectedMetrics((current) => {
      const kept = current.filter((name) => usable.has(name))
      return kept.length > 0 ? kept : available.data!.computable_for_all
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available.data])

  const judgeSelected = selectedMetrics.some(
    (name) => definitions.find((d) => d.name === name)?.kind === 'judge',
  )
  const judgeReady = (judge.data ?? []).some((p: Provider) => p.api_key_set)

  const byFamily = useMemo(() => {
    const groups: Record<string, MetricDefinition[]> = {}
    for (const definition of definitions) {
      ;(groups[definition.family] ??= []).push(definition)
    }
    return groups
  }, [definitions])

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>起一轮</h3>

      <div className="row" style={{ marginBottom: 10 }}>
        <label className="field">
          数据集
          <DatasetPicker
            all={datasetNames}
            selected={selectedDatasets}
            onChange={setSelectedDatasets}
          />
        </label>
      </div>

      <h4>指标（勾了的才算）</h4>
      {available.loading && <Loading what="勾选范围" />}
      {available.data && (
        <>
          {available.data.partial.length > 0 && (
            <div className="note warn small">
              <strong>{available.data.partial.length} 项只对部分数据集成立。</strong>
              勾了不会报错：缺依赖的那组会省略它并写明原因，而不是伪造 0 分。
            </div>
          )}
          {Object.entries(byFamily).map(([family, entries]) => (
            <div key={family} style={{ marginBottom: 10 }}>
              <div className="small muted">{FAMILY_LABELS[family] ?? family}</div>
              <div className="metric-grid">
                {entries.map((definition) => {
                  const usable = available.data!.computable_for_some.includes(definition.name)
                  const partial = available.data!.partial.includes(definition.name)
                  return (
                    <label
                      key={definition.name}
                      className={`check${usable ? '' : ' unavailable'}`}
                      title={
                        usable
                          ? definition.description
                          : `所选数据集都缺 ${definition.requires.join('、')}，这一项算不了`
                      }
                    >
                      <input
                        type="checkbox"
                        disabled={!usable}
                        checked={selectedMetrics.includes(definition.name)}
                        onChange={(event) =>
                          setSelectedMetrics((current) =>
                            event.target.checked
                              ? [...current, definition.name]
                              : current.filter((name) => name !== definition.name),
                          )
                        }
                      />
                      <span>
                        {definition.name}
                        {definition.per_k && <span className="muted">@k</span>}
                        {partial && (
                          <span className="tag warn" style={{ marginLeft: 4 }}>
                            部分
                          </span>
                        )}
                        {!definition.higher_is_better && (
                          <span className="tag" style={{ marginLeft: 4 }} title="越低越好">
                            ↓
                          </span>
                        )}
                      </span>
                    </label>
                  )
                })}
              </div>
            </div>
          ))}
          <div className="row tight small">
            <button
              className="action small"
              onClick={() => setSelectedMetrics(available.data!.computable_for_all)}
            >
              全选（对每组都算得出来）
            </button>
            <button
              className="action small"
              onClick={() => setSelectedMetrics(available.data!.computable_for_some)}
            >
              全选（含仅部分成立的）
            </button>
            <button className="action small" onClick={() => setSelectedMetrics([])}>
              清空
            </button>
            <span className="muted">已选 {selectedMetrics.length} 项</span>
          </div>
        </>
      )}

      {judgeSelected && (
        <div className={`note ${judgeReady ? '' : 'warn'}`}>
          <strong>勾了 judge 类指标。</strong>
          {judgeReady ? (
            <> judge 模型已配置，评测跑完后从「任务」层起一次 judge。它会调模型花钱。</>
          ) : (
            <>
              {' '}
              还没有可用的 judge 模型端点 —— Akasha 只回传 apiKeySet 布尔量、
              从不回传 key，所以这份凭据要单独配。
              <button className="action small" style={{ marginLeft: 8 }} onClick={onOpenSettings}>
                去配置层
              </button>
            </>
          )}
        </div>
      )}

      <h4>标签与 k</h4>
      <div className="row">
        <label className="field">
          查询层标签
          <input value={queryLabel} onChange={(event) => setQueryLabel(event.target.value)} />
        </label>
        <label className="field">
          评测层标签
          <input value={evalLabel} onChange={(event) => setEvalLabel(event.target.value)} />
        </label>
        <label className="field">
          k（逗号分隔）
          <input value={ks} onChange={(event) => setKs(event.target.value)} />
        </label>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <button
          className="action primary"
          disabled={startQuery.busy || selectedDatasets.length === 0}
          onClick={() =>
            startQuery.run(() =>
              api.startTask('query', {
                label: layer.label,
                query_label: queryLabel,
                datasets: selectedDatasets,
              }),
            )
          }
        >
          {startQuery.busy ? '启动中…' : '① 跑查询'}
        </button>
        <button
          className="action"
          disabled={startEval.busy || selectedMetrics.length === 0}
          onClick={() =>
            startEval.run(() =>
              api.startTask('evaluate', {
                query_label: queryLabel,
                eval_label: evalLabel,
                datasets: selectedDatasets,
                metrics: selectedMetrics,
                k: ks
                  .split(',')
                  .map((value) => Number(value.trim()))
                  .filter((value) => Number.isFinite(value) && value > 0),
              }),
            )
          }
        >
          {startEval.busy ? '启动中…' : '② 算指标'}
        </button>
        <span className="small muted">
          查询约 10–14 秒每条；算指标是纯离线的，秒级。进度在「任务」层看。
        </span>
      </div>

      {startQuery.error && <div className="note bad">{startQuery.error}</div>}
      {startEval.error && <div className="note bad">{startEval.error}</div>}
      {(startQuery.result || startEval.result) && (
        <div className="note">
          已启动。到「任务」层看进度；跑完回来刷新。
          <button className="action small" style={{ marginLeft: 8 }} onClick={onDone}>
            刷新
          </button>
        </div>
      )}
    </div>
  )
}

const FAMILY_LABELS: Record<string, string> = {
  retrieval: '检索',
  qa: '答案质量',
  attribution: '引用归因',
  multihop: '多跳',
  judge: 'judge（需要模型，会花钱）',
}

/** 已有的查询层与它们的生成结果。 */
function QueryLayers({
  layers,
  onOpenReport,
}: {
  layers: QueryLayerSummary[]
  onOpenReport: (evalLayerId: number) => void
}) {
  const [open, setOpen] = useState<number | null>(layers[0]?.id ?? null)

  if (layers.length === 0)
    return <Empty>这一层还没有查询层。跑一次查询就会出现在这里。</Empty>

  return (
    <>
      <h3>查询层与生成结果</h3>
      {layers.map((layer) => (
        <div key={layer.id} className="panel">
          <div className="spread">
            <div>
              <strong>#{layer.id}</strong> <span className="tag">{layer.label}</span>{' '}
              {layer.model_configs_match_index === 0 && (
                <span className="tag warn" title="模型配置与入库时不一致">
                  配置漂移
                </span>
              )}
            </div>
            <div className="row tight small muted">
              <span>
                {layer.score_threshold === null
                  ? '服务端默认阈值'
                  : `阈值 ${layer.score_threshold}`}
              </span>
              <span>并发 {layer.concurrency}</span>
              <button
                className="action small"
                onClick={() => setOpen(open === layer.id ? null : layer.id)}
              >
                {open === layer.id ? '收起响应' : '看响应'}
              </button>
            </div>
          </div>

          <div className="row small muted" style={{ marginTop: 4 }}>
            {Object.entries(layer.stats).map(([dataset, stat]) => (
              <span key={dataset}>
                {dataset}: {stat.responses} 条
                {stat.failures > 0 && (
                  <span className="tag bad" style={{ marginLeft: 4 }}>
                    {stat.failures} 失败
                  </span>
                )}
              </span>
            ))}
          </div>

          <div className="row" style={{ marginTop: 8 }}>
            {layer.eval_layers.length === 0 ? (
              <span className="small muted">还没有评测层。</span>
            ) : (
              layer.eval_layers.map((entry) => (
                <button
                  key={entry.id}
                  className="action small"
                  onClick={() => onOpenReport(entry.id)}
                >
                  报告：#{entry.id} {entry.label}
                </button>
              ))
            )}
          </div>

          {open === layer.id && <Responses queryLayerId={layer.id} />}
        </div>
      ))}
    </>
  )
}

function Responses({ queryLayerId }: { queryLayerId: number }) {
  const [offset, setOffset] = useState(0)
  const [mode, setMode] = useState('')
  const [openSample, setOpenSample] = useState<string | null>(null)
  const limit = 20

  const { data, error, loading } = useAsync(
    () => api.responses(queryLayerId, { answer_mode: mode || undefined, limit, offset }),
    [queryLayerId, mode, offset],
  )

  return (
    <div style={{ marginTop: 12 }}>
      {loading && <Loading what="响应" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          <div className="row tight small">
            <span className="muted">按 answerMode：</span>
            <button
              className={`action small${mode === '' ? ' primary' : ''}`}
              onClick={() => {
                setMode('')
                setOffset(0)
              }}
            >
              全部 {data.total}
            </button>
            {Object.entries(data.count_by_answer_mode).map(([name, count]) => (
              <button
                key={name}
                className={`action small${mode === name ? ' primary' : ''}`}
                onClick={() => {
                  setMode(name)
                  setOffset(0)
                }}
              >
                {name} {count}
              </button>
            ))}
          </div>
          <p className="small muted" style={{ marginTop: 6 }}>
            <code>no_match</code> 与 <code>general</code> 无条件返回空 retrievedSources，
            它们的检索得分按定义为 0 —— 那是生成端拒答，不是检索失败。
          </p>

          <table>
            <thead>
              <tr>
                <th>sample_id</th>
                <th>问题</th>
                <th>模式</th>
                <th className="num">召回</th>
                <th className="num">引用</th>
                <th className="num">延迟</th>
              </tr>
            </thead>
            <tbody>
              {data.responses.map((row) => (
                <tr
                  key={row.sample_id}
                  className={`clickable${openSample === row.sample_id ? ' selected' : ''}`}
                  onClick={() =>
                    setOpenSample(openSample === row.sample_id ? null : row.sample_id)
                  }
                >
                  <td className="small mono truncate">{row.sample_id}</td>
                  <td className="small truncate">{row.question}</td>
                  <td>
                    <ModeTag mode={row.answer_mode} />
                    {row.http_status >= 300 && (
                      <span className="tag bad" style={{ marginLeft: 4 }}>
                        HTTP {row.http_status}
                      </span>
                    )}
                  </td>
                  <td className="num">{row.retrieved_count}</td>
                  <td className="num">{row.citation_count}</td>
                  <td className="num">{row.latency_ms ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}

      {openSample && <ResponseBody queryLayerId={queryLayerId} sampleId={openSample} />}
    </div>
  )
}

function ResponseBody({
  queryLayerId,
  sampleId,
}: {
  queryLayerId: number
  sampleId: string
}) {
  const { data, error, loading } = useAsync(
    () => api.response(queryLayerId, sampleId),
    [queryLayerId, sampleId],
  )

  if (loading) return <Loading what="响应体" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const body = data.response as Record<string, unknown> | null
  const answer = (body?.answer as string) ?? ''

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <h4 style={{ marginTop: 0 }}>{sampleId}</h4>
      <dl className="kv">
        <dt>问题</dt>
        <dd>{data.question as string}</dd>
        <dt>生成的答案</dt>
        <dd>{answer || <span className="muted">（空）</span>}</dd>
      </dl>
      <p className="small muted">
        存的是<strong>完整响应体</strong>，不是当下用得到的那几个字段 ——
        重跑一次要烧 LLM 调用。
      </p>
      <pre className="block tall">{JSON.stringify(body, null, 2)}</pre>
      {data.audit ? (
        <>
          <h4>审计记录（retrievalDiagnostics）</h4>
          <p className="small muted">
            它不在 HTTP 响应里，controller 解构时排除了 —— 只写进
            <code>knowledge_query_audit.metadata</code>，这份是抄进库的存档。
          </p>
          <pre className="block">{JSON.stringify(data.audit, null, 2)}</pre>
        </>
      ) : (
        <p className="small muted">
          没有审计记录。要它得跑一次「审计归因」，且需要只读 Postgres 连接。
        </p>
      )}
    </div>
  )
}


function QuickCheck({ onOpenTasks }: { onOpenTasks: () => void }) {
  const datasets = useAsync(() => api.datasets(), [])
  const [dataset, setDataset] = useState('')
  const [samples, setSamples] = useState(3)
  const start = useAction<{ id: number }>()
  const available = (datasets.data ?? []).filter((d) =>
    ['hotpotqa', '2wikimultihopqa', 'musique'].includes(d.name),
  )
  const selected = dataset || available[0]?.name || ''

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>小样本验证</h3>
      <p className="small muted">
        自动抽样、入库编译、查询并生成报告，检查响应结构、来源映射与断点续跑。
        会调用模型；每篇 gold 配一篇干扰文档，最多导入 50 篇。独立创建的编译层与远端 Space 会保留，便于复查。
      </p>
      <div className="row">
        <label className="field">
          数据集
          <select value={selected} onChange={(event) => setDataset(event.target.value)}>
            {available.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
          </select>
        </label>
        <label className="field">
          样本数（1–5）
          <input type="number" min={1} max={5} value={samples}
            onChange={(event) => setSamples(Number(event.target.value))} />
        </label>
        <button className="action primary"
          disabled={start.busy || !!start.result || !selected || !Number.isInteger(samples) || samples < 1 || samples > 5}
          onClick={() => start.run(() => api.startTask('verify', { dataset: selected, samples }))}>
          {start.busy ? '启动中…' : '开始验证'}
        </button>
      </div>
      {datasets.error && <Failed error={datasets.error} />}
      {!datasets.loading && !datasets.error && available.length === 0 &&
        <p className="muted">请先在归一化页面准备带 gold 文档的数据集。</p>}
      {start.error && <div className="note bad">{start.error}</div>}
      {start.result && <div className="note">
        验证任务 #{start.result.id} 已启动。完成后可在报告层查看结果。
        <button className="action small" onClick={onOpenTasks}>查看进度与检查结果</button>
      </div>}
    </div>
  )
}
