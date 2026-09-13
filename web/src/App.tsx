import { useState } from 'react'
import { Badcase } from './views/Badcase'
import { Compile } from './views/Compile'
import { Datasets } from './views/Datasets'
import { Evaluate } from './views/Evaluate'
import { Normalize } from './views/Normalize'
import { Query } from './views/Query'
import { Testing } from './views/Testing'
import { Settings } from './views/Settings'
import { Tasks } from './views/Tasks'

const LAYERS = [
  { key: 'datasets', step: '1', label: '数据集', hint: '原始样例' },
  { key: 'normalize', step: '2', label: '归一化', hint: '适配器与归一化后的样本' },
  { key: 'compile', step: '3', label: '编译层', hint: '编译模型与文档变化' },
  { key: 'query', step: '4', label: '查询层', hint: '运行查询、查看检索与生成响应' },
  { key: 'evaluate', step: '5', label: '评测层', hint: '选择指标、计算并查看评测结果' },
  { key: 'badcase', step: '6', label: '归因层', hint: '完整链路与根因' },
] as const

const CROSS_CUTTING = [
  { key: 'tasks', label: '任务', hint: '各阶段任务的启动、停止与清理' },
  { key: 'settings', label: '配置', hint: 'Akasha 连接与模型端点' },
  { key: 'testing', label: '测试', hint: '小样本链路测试' },
] as const

type ViewKey = (typeof LAYERS)[number]['key'] | (typeof CROSS_CUTTING)[number]['key']

export function App() {
  const [view, setView] = useState<ViewKey>('datasets')
  const [activeEval, setActiveEval] = useState<number | null>(null)
  const [activeQuery, setActiveQuery] = useState<number | null>(null)
  const [activeIndexLayer, setActiveIndexLayer] = useState<number | null>(null)
  const selectIndex = (id: number) => {
    setActiveIndexLayer(id)
    setActiveQuery(null)
    setActiveEval(null)
  }
  const selectQuery = (id: number) => {
    setActiveQuery(id)
    setActiveEval(null)
  }

  const openBadcase = (evalLayerId: number) => {
    setActiveEval(evalLayerId)
    setView('badcase')
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Akasha-Benchmark</h1>
        <div className="sub">观测与评测平台</div>

        <nav className="nav">
          {LAYERS.map((layer) => (
            <button
              key={layer.key}
              className={view === layer.key ? 'active' : ''}
              title={layer.hint}
              onClick={() => setView(layer.key)}
            >
              <span className="step">{layer.step}</span>
              {layer.label}
            </button>
          ))}
        </nav>

        <div style={{ height: 14 }} />
        <nav className="nav">
          {CROSS_CUTTING.map((entry) => (
            <button
              key={entry.key}
              className={view === entry.key ? 'active' : ''}
              title={entry.hint}
              onClick={() => setView(entry.key)}
            >
              <span className="step">·</span>
              {entry.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        {view === 'datasets' && <Datasets onOpenTasks={() => setView('tasks')} />}
        {view === 'normalize' && <Normalize onOpenTasks={() => setView('tasks')} />}
        {view === 'compile' && (
          <Compile
            activeLayer={activeIndexLayer}
            onSelectLayer={selectIndex}
            onOpenQuery={(id) => { selectIndex(id); setView('query') }}
            onOpenTasks={() => setView('tasks')}
          />
        )}
        {view === 'query' && (
          <Query activeLayer={activeIndexLayer} onSelectLayer={selectIndex}
            onEvaluate={(indexId, queryId) => {
              setActiveIndexLayer(indexId)
              selectQuery(queryId)
              setView('evaluate')
            }}
            onOpenSettings={() => setView('settings')} onOpenTasks={() => setView('tasks')} />
        )}
        {view === 'evaluate' && (
          <Evaluate activeLayer={activeIndexLayer} activeQuery={activeQuery} activeEval={activeEval}
            onSelectLayer={selectIndex} onSelectQuery={selectQuery} onSelectEval={setActiveEval}
            onOpenBadcase={openBadcase} onOpenSettings={() => setView('settings')}
            onOpenTasks={() => setView('tasks')} />
        )}
        {view === 'badcase' && (
          <Badcase
            activeEval={activeEval}
            onSelectEval={setActiveEval}
            onOpenSettings={() => setView('settings')}
          />
        )}
        {view === 'testing' && <Testing onOpenTasks={() => setView('tasks')} />}
        {view === 'tasks' && <Tasks />}
        {view === 'settings' && <Settings />}
      </main>
    </div>
  )
}
