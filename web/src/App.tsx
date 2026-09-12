import { useState } from 'react'
import { api } from './api'
import { useAsync } from './ui'
import { Badcase } from './views/Badcase'
import { Compile } from './views/Compile'
import { Datasets } from './views/Datasets'
import { Evaluate } from './views/Evaluate'
import { Normalize } from './views/Normalize'
import { Report } from './views/Report'
import { Settings } from './views/Settings'
import { Tasks } from './views/Tasks'

/** 左栏八层，顺序即流水线顺序。
 *
 * 数据集 -> 归一化 -> 编译 -> 评测 -> 报告 -> 归因，六层是一条链；
 * 任务与配置是横切的，所以放在下面另一组。
 *
 * 「指标」不单列一层：它是评测层的可勾选配置，而勾选范围由所选数据集的
 * 标注决定 —— 单列会让人以为它是数据集的属性，那正好是反过来的。
 */
const LAYERS = [
  { key: 'datasets', step: '1', label: '数据集', hint: '原始样例' },
  { key: 'normalize', step: '2', label: '归一化', hint: '适配器与归一化后的样本' },
  { key: 'compile', step: '3', label: '编译层', hint: '编译模型与文档变化' },
  { key: 'evaluate', step: '4', label: '评测层', hint: '勾指标、配模型、看响应' },
  { key: 'report', step: '5', label: '报告层', hint: '本轮指标结果' },
  { key: 'badcase', step: '6', label: '归因层', hint: '完整链路与根因' },
] as const

const CROSS_CUTTING = [
  { key: 'tasks', label: '任务', hint: '各阶段任务的启动、停止与清理' },
  { key: 'settings', label: '配置', hint: 'Akasha 连接与模型端点' },
] as const

type ViewKey = (typeof LAYERS)[number]['key'] | (typeof CROSS_CUTTING)[number]['key']

/** 当前连的是哪个 Akasha。所有在线阶段都跑在它上面，值得一直显示在视野里。 */
function CurrentConnection({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { data } = useAsync(() => api.connection(), [])
  if (!data) return null

  return (
    <div className="clickable" onClick={onOpenSettings} title={`${data.email} @ ${data.base_url}`}>
      <span className="muted">akasha</span>{' '}
      <strong className="mono">{data.base_url.replace(/^https?:\/\//, '')}</strong>
      {!data.password && (
        <span className="tag bad" style={{ marginLeft: 4 }}>
          未配密码
        </span>
      )}
      {data.last_check_ok === 0 && (
        <span className="tag warn" style={{ marginLeft: 4 }}>
          上次测试失败
        </span>
      )}
    </div>
  )
}

export function App() {
  const [view, setView] = useState<ViewKey>('datasets')
  // 跨层携带的选择。评测层选了哪个查询层、报告层看哪个评测层，
  // 换页时要记住 —— 否则每次切回来都要重选一遍。
  const [activeEval, setActiveEval] = useState<number | null>(null)
  const [activeIndexLayer, setActiveIndexLayer] = useState<number | null>(null)

  const openReport = (evalLayerId: number) => {
    setActiveEval(evalLayerId)
    setView('report')
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

        <div className="foot">
          <CurrentConnection onOpenSettings={() => setView('settings')} />
          {activeEval !== null && <div>评测层 #{activeEval}</div>}
          {activeIndexLayer !== null && <div>编译层 #{activeIndexLayer}</div>}
        </div>
      </aside>

      <main className="main">
        {view === 'datasets' && <Datasets />}
        {view === 'normalize' && <Normalize />}
        {view === 'compile' && (
          <Compile
            activeLayer={activeIndexLayer}
            onSelectLayer={setActiveIndexLayer}
            onOpenTasks={() => setView('tasks')}
            onOpenSettings={() => setView('settings')}
          />
        )}
        {view === 'evaluate' && (
          <Evaluate
            activeLayer={activeIndexLayer}
            onSelectLayer={setActiveIndexLayer}
            onOpenReport={openReport}
            onOpenSettings={() => setView('settings')}
            onOpenTasks={() => setView('tasks')}
          />
        )}
        {view === 'report' && (
          <Report
            activeEval={activeEval}
            onSelectEval={setActiveEval}
            onOpenBadcase={openBadcase}
          />
        )}
        {view === 'badcase' && (
          <Badcase
            activeEval={activeEval}
            onSelectEval={setActiveEval}
            onOpenSettings={() => setView('settings')}
          />
        )}
        {view === 'tasks' && <Tasks />}
        {view === 'settings' && <Settings />}
      </main>
    </div>
  )
}
