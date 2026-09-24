import { useState } from 'react'
import { api } from '../api'
import type { MetricsView, Provider, Sample } from '../types'
import { Failed, Field, Loading, useAction, useAsync } from '../ui'

const DATASETS = ['hotpotqa', '2wikimultihopqa', 'musique']

export function Testing({ onOpenTasks }: { onOpenTasks: () => void }) {
  const datasets = useAsync(() => api.datasets(), [])
  const judges = useAsync<Provider[]>(() => api.providers('judge'), [])
  const metrics = useAsync<MetricsView>(() => api.metrics(DATASETS), [])
  const [dataset, setDataset] = useState(DATASETS[0]!)
  const [selected, setSelected] = useState<Sample | null>(null)
  const [withJudge, setWithJudge] = useState(false)
  const [useModel, setUseModel] = useState(false)
  const start = useAction<unknown>()

  if (datasets.loading) return <Loading what="数据集状态" />
  if (datasets.error) return <Failed error={datasets.error} />

  const normalized = (datasets.data?.datasets ?? [])
    .filter((d) => d.normalized && DATASETS.includes(d.name))
    .map((d) => d.name)
  const activeDataset = normalized.includes(dataset) ? dataset : normalized[0] ?? ''
  const computable = new Set(metrics.data?.per_dataset[activeDataset] ?? [])
  const metricCount = (metrics.data?.definitions ?? []).filter(
    (definition) =>
      computable.has(definition.name) && (withJudge || definition.kind === 'deterministic'),
  ).length

  return (
    <>
      <h2>测试</h2>

      {start.error && <Failed error={start.error} />}

      {normalized.length === 0 ? (
        <div className="note warn">
          链路测试需要一个已归一化的数据集（{DATASETS.join('、')}）。请先在「归一化」页处理。
        </div>
      ) : (
        <div className="panel">
          <div className="row">
            <Field label="数据集" hint="只用有 gold 标注的三组">
              <select
                value={activeDataset}
                onChange={(e) => {
                  setDataset(e.target.value)
                  setSelected(null)
                }}
              >
                {normalized.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>
          </div>

          <SamplePicker
            dataset={activeDataset}
            selected={selected}
            onSelect={setSelected}
          />

          <div className="row tight" style={{ marginTop: 8 }}>
            <label className="check">
              <input
                type="checkbox"
                checked={withJudge}
                disabled={(judges.data ?? []).length === 0}
                onChange={() => setWithJudge(!withJudge)}
              />
              包含评估模型
            </label>
            <label className="check">
              <input type="checkbox" checked={useModel} onChange={() => setUseModel(!useModel)} />
              包含归因模型
            </label>
          </div>

          <div className="panel-actions">
            <button
              className="action primary"
              disabled={start.busy || !selected}
              onClick={() =>
                start.run(async () => {
                  const task = await api.startChain({
                    dataset: activeDataset,
                    sample_id: selected!.sample_id,
                    use_model: useModel,
                    with_judge: withJudge,
                  })
                  onOpenTasks()
                  return task
                })
              }
            >
              {start.busy ? '启动中…' : '开始链路测试'}
            </button>
            <span className="small muted">
              仅对所选问题执行编译 → 查询 → 评测 → 归因；目标语料 5 篇，固定 k=2；自动全选 {metricCount} 个
              {withJudge ? '可计算指标（含 Judge）' : '确定性指标'}
            </span>
          </div>
        </div>
      )}

    </>
  )
}

function SamplePicker({
  dataset,
  selected,
  onSelect,
}: {
  dataset: string
  selected: Sample | null
  onSelect: (sample: Sample) => void
}) {
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const results = useAsync(
    () => api.samples(dataset, { q: q || undefined, limit: 20 }),
    [dataset, q],
  )

  return (
    <div className="sample-picker">
      <div className="row">
        <Field label="搜索样本" hint="匹配问题、答案、gold 内容标题或正文">
          <div className="row tight">
            <input
              value={term}
              onChange={(event) => setTerm(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') setQ(term.trim())
              }}
              placeholder="输入内容标题或关键词"
            />
            <button className="action" onClick={() => setQ(term.trim())}>搜索</button>
          </div>
        </Field>
      </div>

      {selected && (
        <div className="note ok small">
          已选择问题 <strong>{selected.question}</strong>
          <span className="mono" style={{ marginLeft: 8 }}>{selected.sample_id}</span>
          {selected.gold_titles?.length ? (
            <div className="small muted" style={{ marginTop: 4 }}>
              gold 文档：{selected.gold_titles.join(' / ')}；测试语料共补到 5 篇
            </div>
          ) : null}
        </div>
      )}

      {results.loading && <Loading what="测试样本" />}
      {results.error && <Failed error={results.error} />}
      {results.data && (
        <div className="sample-picker-results">
          <div className="small muted">
            匹配 {results.data.total} 条，显示前 {results.data.samples.length} 条
          </div>
          {results.data.samples.map((sample) => (
            <button
              key={sample.sample_id}
              type="button"
              className={`sample-picker-item ${selected?.sample_id === sample.sample_id ? 'selected' : ''}`}
              onClick={() => onSelect(sample)}
            >
              <span className="sample-picker-title">{sample.question}</span>
              <span className="small muted">
                gold 文档：{sample.gold_titles?.join(' / ') || '（无内容标题）'}
              </span>
              <span className="small muted">参考答案：{sample.answers.join(' / ')}</span>
              <span className="mono small muted">{sample.sample_id}</span>
            </button>
          ))}
          {results.data.total === 0 && <div className="note plain">没有匹配样本。</div>}
        </div>
      )}
    </div>
  )
}
