import { api } from '../api'
import { Empty, Failed, Loading, useAsync } from '../ui'
import { EvaluationForm } from './EvaluationForm'
import { EvaluationResults } from './EvaluationResults'

export function Evaluate({ activeLayer, activeQuery, activeEval, onSelectLayer, onSelectQuery, onSelectEval, onOpenBadcase, onOpenSettings, onOpenTasks }: {
  activeLayer: number | null; activeQuery: number | null; activeEval: number | null
  onSelectLayer: (id: number) => void; onSelectQuery: (id: number) => void
  onSelectEval: (id: number) => void; onOpenBadcase: (id: number) => void
  onOpenSettings: () => void; onOpenTasks: () => void
}) {
  const layers = useAsync(() => api.layers(), [])
  const definitions = useAsync(() => api.metricDefinitions(), [])
  const all = layers.data?.index_layers ?? []
  const layer = all.find((entry) => entry.id === activeLayer) ?? all[0]
  const queries = layer?.query_layers.filter((entry) => entry.finished_at && Object.keys(entry.stats).length > 0) ?? []
  const query = queries.find((entry) => entry.id === activeQuery) ?? queries[0]
  const evaluation = query?.eval_layers.find((entry) => entry.id === activeEval) ?? query?.eval_layers[0]
  if (layers.loading || definitions.loading) return <Loading what="评测层" />
  if (layers.error || definitions.error) return <Failed error={layers.error || definitions.error!} />
  return <>
    <div className="spread"><h2>评测层</h2><button className="action small" onClick={layers.reload}>刷新</button></div>
    {!layer ? <Empty>暂无编译批次。</Empty> : <>
      <div className="row" style={{ margin: '12px 0' }}>
        <label className="field">编译批次（run_id）<select value={layer.id} onChange={(event) => onSelectLayer(Number(event.target.value))}>
          {all.map((entry) => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
        </select></label>
        {query && <label className="field">查询记录<select value={query.id} onChange={(event) => onSelectQuery(Number(event.target.value))}>
          {queries.map((entry) => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
        </select></label>}
      </div>
      {!query ? <Empty>此编译批次暂无已结束的查询记录。</Empty> : <>
        <EvaluationForm key={query.id} layer={layer} query={query} definitions={definitions.data ?? []} onOpenSettings={onOpenSettings} onOpenTasks={onOpenTasks} />
        <h3>评测结果</h3>
        {query.eval_layers.length === 0 ? <Empty>暂无评测结果。</Empty> : <>
          <div className="row" style={{ marginBottom: 12 }}>{query.eval_layers.map((entry) => <button key={entry.id}
            className={`action small${evaluation?.id === entry.id ? ' primary' : ''}`} onClick={() => onSelectEval(entry.id)}>{entry.label}</button>)}</div>
          {evaluation && <>
            <p className="small mono">{evaluation.run_id} → {query.label} → {evaluation.label} · {evaluation.finished_at ? '已完成' : '未完成，进度见任务'}</p>
            <EvaluationResults key={evaluation.id} evalLayerId={evaluation.id} onOpenBadcase={onOpenBadcase} />
          </>}
        </>}
      </>}
    </>}
  </>
}
