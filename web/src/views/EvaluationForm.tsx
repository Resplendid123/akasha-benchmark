import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { IndexLayer, MetricDefinition, Provider, QueryLayerSummary } from '../types'
import { DatasetPicker, Failed, Loading, useAction, useAsync } from '../ui'

export function EvaluationForm({
  layer,
  definitions,
  onOpenSettings,
  onOpenTasks,
  query,
}: {
  layer: IndexLayer
  definitions: MetricDefinition[]
  onOpenSettings: () => void
  onOpenTasks: () => void
  query: QueryLayerSummary
}) {
  const datasetNames = useMemo(() => Object.keys(query.stats), [query])
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>(datasetNames)
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [ks, setKs] = useState('2,5,10')
  const [evalLabel, setEvalLabel] = useState(`${query.label}-eval${String(query.eval_layers.length + 1).padStart(3, '0')}`)

  const available = useAsync(
    () =>
      selectedDatasets.length > 0
        ? api.availableMetrics(selectedDatasets)
        : Promise.resolve(null),
    [selectedDatasets.join(',')],
  )
  const judge = useAsync(() => api.providers('judge'), [])
  const startEval = useAction<{ id: number }>()

  // 数据集换了，勾选范围跟着变 —— 把已经不可用的指标去掉。
  useEffect(() => {
    if (!available.data) return
    const usable = new Set(available.data.computable_for_some)
    setSelectedMetrics((current) => {
      const kept = current.filter((name) => usable.has(name))
      return kept.length > 0 ? kept : available.data!.computable_for_all.filter(
        (name) => definitions.find((entry) => entry.name === name)?.kind !== 'judge',
      )
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available.data])

  const judgeSelected = selectedMetrics.some(
    (name) => definitions.find((d) => d.name === name)?.kind === 'judge',
  )
  const [provider, setProvider] = useState('')
  const providers = (judge.data ?? []).filter((p: Provider) => p.api_key_set)
  const selectedProvider = provider || providers[0]?.label || ''
  const judgeReady = !!selectedProvider
  const kValues = ks.split(',').map((value) => Number(value.trim()))
  const validK = kValues.length > 0 && kValues.every((value) => Number.isInteger(value) && value > 0)

  const byFamily = useMemo(() => {
    const groups: Record<string, MetricDefinition[]> = {}
    for (const definition of definitions) {
      ;(groups[definition.family] ??= []).push(definition)
    }
    return groups
  }, [definitions])

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>评测参数</h3>
      <p className="small mono">{layer.label} → {query.label}</p>

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
        <div className="row" style={{ margin: '12px 0' }}>
          <label className="field">Judge 模型
            <select value={selectedProvider} onChange={(event) => setProvider(event.target.value)}>
              {providers.map((entry) => <option key={entry.id} value={entry.label}>{entry.label} · {entry.model}</option>)}
            </select>
          </label>
          {!judgeReady && <button className="action small" onClick={onOpenSettings}>配置 Judge 模型</button>}
          <span className="small muted">所选 Judge 指标会调用模型。</span>
        </div>
      )}
      {available.error && <Failed error={available.error} />}
      {judge.error && <Failed error={judge.error} />}

      <h4>标签与 k</h4>
      <div className="row">
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
        <button className="action primary"
          disabled={startEval.busy || !!startEval.result || !evalLabel.trim() || !validK || selectedMetrics.length === 0 || selectedDatasets.length === 0 || available.loading || !!available.error || (judgeSelected && !judgeReady) || query.eval_layers.some((entry) => entry.label === evalLabel.trim())}
          onClick={() => startEval.run(() => api.startTask('evaluate', {
            query_label: query.label,
            eval_label: evalLabel.trim(),
            datasets: selectedDatasets,
            metrics: selectedMetrics,
            k: kValues,
            ...(judgeSelected ? { provider_label: selectedProvider } : {}),
          }))}
        >{startEval.busy ? '启动中…' : '开始评测'}</button>
      </div>
      {startEval.error && <Failed error={startEval.error} />}
      {startEval.result && <div className="note">评测任务 #{startEval.result.id} 已启动。
        <button className="action small" onClick={onOpenTasks}>查看任务</button>
      </div>}
    </div>
  )
}

const FAMILY_LABELS: Record<string, string> = {
  retrieval: '检索', qa: '答案质量', attribution: '引用归因', multihop: '多跳', judge: 'Judge',
}
