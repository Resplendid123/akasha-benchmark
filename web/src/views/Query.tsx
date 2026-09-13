import { useState } from 'react'
import { api } from '../api'
import type { IndexLayer } from '../types'
import { DatasetPicker, Empty, Failed, Loading, useAction, useAsync } from '../ui'
import { Responses } from './Responses'

export function Query({ activeLayer, onSelectLayer, onEvaluate, onOpenTasks, onOpenSettings }: {
  activeLayer: number | null
  onSelectLayer: (id: number) => void
  onEvaluate: (indexId: number, queryId: number) => void
  onOpenTasks: () => void
  onOpenSettings: () => void
}) {
  const layers = useAsync(() => api.layers(), [])
  const all = layers.data?.index_layers ?? []
  const layer = all.find((entry) => entry.id === activeLayer) ?? all[0]
  if (layers.loading) return <Loading what="查询层" />
  if (layers.error) return <Failed error={layers.error} />
  return <>
    <div className="spread"><h2>查询层</h2><button className="action small" onClick={layers.reload}>刷新</button></div>
    {!layer ? <Empty>暂无编译批次。</Empty> : <>
      <label className="field" style={{ margin: '12px 0' }}>编译批次（run_id）
        <select value={layer.id} onChange={(event) => onSelectLayer(Number(event.target.value))}>
          {all.map((entry) => <option key={entry.id} value={entry.id}>{entry.run_id}{entry.ready_for_query ? '' : ' · 未就绪'}</option>)}
        </select>
      </label>
      <QueryForm key={layer.id} layer={layer} onOpenTasks={onOpenTasks} onOpenSettings={onOpenSettings} />
      <h3>查询记录</h3>
      {layer.query_layers.length === 0 && <Empty>暂无查询记录。</Empty>}
      {layer.query_layers.map((query) => <div className="panel" key={query.id}>
        <div className="spread">
          <strong>{query.run_id} → {query.label}</strong>
          <button className="action small" disabled={!query.finished_at || Object.keys(query.stats).length === 0}
            onClick={() => onEvaluate(layer.id, query.id)}>去评测</button>
        </div>
        <div className="row small muted" style={{ margin: '8px 0' }}>
          <span>查询 #{query.id}</span><span>计划 {Object.values(query.selection ?? {}).reduce((sum, ids) => sum + ids.length, 0)} 条</span><span>{query.finished_at ? '已结束' : '未完成'}</span>
          {Object.entries(query.stats).map(([name, stats]) => <span key={name}>{name}：{stats.responses} 条，{stats.failures} 失败</span>)}
        </div>
        <details><summary>查看查询与响应</summary><Responses key={query.id} queryLayerId={query.id} /></details>
      </div>)}
    </>}
  </>
}

function QueryForm({ layer, onOpenTasks, onOpenSettings }: {
  layer: IndexLayer; onOpenTasks: () => void; onOpenSettings: () => void
}) {
  const [datasets, setDatasets] = useState(layer.datasets.map((entry) => entry.dataset))
  const [count, setCount] = useState(Math.min(10, ...layer.datasets.map((entry) => entry.qa_count)))
  const [label, setLabel] = useState(`${layer.label}-query${String(layer.query_layers.length + 1).padStart(3, '0')}`)
  const [allowDrift, setAllowDrift] = useState(false)
  const start = useAction<{ id: number }>()
  const selected = layer.datasets.filter((entry) => datasets.includes(entry.dataset))
  const maximum = selected.length ? Math.min(...selected.map((entry) => entry.qa_count)) : 0
  const valid = Number.isInteger(count) && count > 0 && count <= maximum
  return <div className="panel">
    <div className="spread"><h3 style={{ margin: 0 }}>查询参数</h3><button className="action small" onClick={onOpenSettings}>问答模型配置</button></div>
    <div className="row" style={{ margin: '12px 0' }}>
      <label className="field">查询记录名称<input value={label} onChange={(event) => setLabel(event.target.value)} /></label>
      <label className="field">每个数据集的 Q 数量<input type="number" min={1} max={maximum} value={count} onChange={(event) => setCount(Number(event.target.value))} /></label>
      <label className="field">数据集<DatasetPicker all={layer.datasets.map((entry) => entry.dataset)} selected={datasets} onChange={setDatasets} /></label>
    </div>
    <div className="small muted">可选上限 {maximum} 条/数据集，本次共 {valid ? count * selected.length : 0} 条。</div>
    <label className="check" style={{ margin: '10px 0' }}><input type="checkbox" checked={allowDrift} onChange={(event) => setAllowDrift(event.target.checked)} />允许使用与编译时不同的问答模型</label>
    {!layer.ready_for_query && <div className="note warn">{layer.not_ready_reasons.join('；')}</div>}
    <button className="action primary" disabled={!layer.ready_for_query || start.busy || !!start.result || !valid || !label.trim() || layer.query_layers.some((entry) => entry.label === label.trim())}
      onClick={() => start.run(() => api.startTask('query', { label: layer.label, query_label: label.trim(), datasets, limit: count, allow_config_drift: allowDrift }))}>
      {start.busy ? '启动中…' : '开始查询'}
    </button>
    {start.error && <Failed error={start.error} />}
    {start.result && <div className="note">查询任务 #{start.result.id} 已启动。<button className="action small" onClick={onOpenTasks}>查看任务</button></div>}
  </div>
}
