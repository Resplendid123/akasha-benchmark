import { useState } from 'react'
import { api } from '../api'
import { Failed, useAction, useAsync } from '../ui'

export function Testing({ onOpenTasks }: { onOpenTasks: () => void }) {
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
      <h2>测试</h2>
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
          {start.busy ? '启动中…' : '开始测试'}
        </button>
      </div>
      {datasets.error && <Failed error={datasets.error} />}
      {!datasets.loading && !datasets.error && available.length === 0 &&
        <p className="muted">请先在归一化页面准备带 gold 文档的数据集。</p>}
      {start.error && <div className="note bad">{start.error}</div>}
      {start.result && <div className="note">
        验证任务 #{start.result.id} 已启动。完成后可在评测层查看结果。
        <button className="action small" onClick={onOpenTasks}>查看进度与检查结果</button>
      </div>}
    </div>
  )
}
