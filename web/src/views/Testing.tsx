import { useState } from 'react'
import { api } from '../api'
import type { Provider } from '../types'
import { Failed, Field, Loading, useAction, useAsync } from '../ui'

const DATASETS = ['hotpotqa', '2wikimultihopqa', 'musique']

/** 测试层：轻量的一次完整六层链路。 */
export function Testing({ onOpenTasks }: { onOpenTasks: () => void }) {
  const datasets = useAsync(() => api.datasets(), [])
  const judges = useAsync<Provider[]>(() => api.providers('judge'), [])
  const [dataset, setDataset] = useState(DATASETS[0]!)
  const [samples, setSamples] = useState(2)
  const [withJudge, setWithJudge] = useState(false)
  const [useModel, setUseModel] = useState(false)
  const start = useAction<unknown>()

  if (datasets.loading) return <Loading what="数据集状态" />
  if (datasets.error) return <Failed error={datasets.error} />

  const normalized = (datasets.data?.datasets ?? [])
    .filter((d) => d.normalized && DATASETS.includes(d.name))
    .map((d) => d.name)

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
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
                {normalized.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="样本数" hint="1–3">
              <input
                type="number"
                min={1}
                max={3}
                value={samples}
                onChange={(e) => setSamples(Number(e.target.value))}
              />
            </Field>
          </div>

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
              disabled={start.busy}
              onClick={() =>
                start.run(async () => {
                  const task = await api.startChain({
                    dataset,
                    samples,
                    use_model: useModel,
                    metrics: withJudge
                      ? ['recall', 'hit', 'em', 'f1', 'citation_recall', 'faithfulness']
                      : ['recall', 'hit', 'em', 'f1', 'citation_recall'],
                  })
                  onOpenTasks()
                  return task
                })
              }
            >
              {start.busy ? '启动中…' : '开始链路测试'}
            </button>
            <span className="small muted">
              编译 → 查询 → 评测 → 归因，四条任务依次跑
            </span>
          </div>
        </div>
      )}

    </>
  )
}
