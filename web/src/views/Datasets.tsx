import { useState } from 'react'
import { api } from '../api'
import type { DatasetEntry } from '../types'
import { DatasetPicker, Failed, Loading, Pager, RecordSearch, num, useAction, useAsync } from '../ui'

export function Datasets({ onOpenTasks }: { onOpenTasks: () => void }) {
  const { data, error, loading, reload } = useAsync(() => api.datasets(), [])
  const [selected, setSelected] = useState<string[]>([])
  const [preview, setPreview] = useState<{ dataset: string; kind: 'qa' | 'corpus' } | null>(null)
  const [reopened, setReopened] = useState(false)
  const start = useAction<unknown>()

  if (loading) return <Loading what="数据集状态" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const names = data.datasets.filter((d) => d.downloadable).map((d) => d.name)
  const targets = selected.length ? selected : names
  // 全部就绪时收起下载栏，由「重新下载」显式打开。
  const allReady = data.datasets.every((d) => d.files_ready)

  return (
    <>
      <div className="spread">
        <h2>数据集</h2>
        <div className="row tight">
          {allReady && !reopened && (
            <button className="action small" onClick={() => setReopened(true)}>
              重新下载
            </button>
          )}
          <button className="action small" onClick={reload}>
            刷新
          </button>
        </div>
      </div>

      {start.error && <Failed error={start.error} />}

      {(!allReady || reopened) && (
        <div className="panel">
          <div className="row">
            <DatasetPicker all={names} selected={selected} onChange={setSelected} />
            <button
              className="action primary"
              disabled={start.busy || !targets.length}
              onClick={() =>
                start.run(async () => {
                  const task = await api.startTask('download', { datasets: targets })
                  onOpenTasks()
                  return task
                })
              }
            >
              {start.busy ? '启动中…' : `下载并校验（${targets.length} 组）`}
            </button>
            {reopened && (
              <button className="action small" onClick={() => setReopened(false)}>
                收起
              </button>
            )}
          </div>
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>数据集</th>
            <th>文件</th>
            <th>状态</th>
            <th className="num">体积</th>
            <th className="num">行数</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.datasets.flatMap((entry) =>
            entry.files.map((file, index) => (
              <tr key={file.file}>
                {index === 0 && <td rowSpan={entry.files.length}>{entry.name}</td>}
                <td className="small mono">{file.file}</td>
                <td>
                  {!file.present ? (
                    <span className="tag bad">
                      {entry.downloadable ? '缺失' : '缺失（本地数据集）'}
                    </span>
                  ) : file.error ? (
                    <span className="tag bad" title={file.error}>
                      解析失败
                    </span>
                  ) : (
                    <span className="tag ok">就绪</span>
                  )}
                </td>
                <td className="num">
                  {file.size_bytes ? `${(file.size_bytes / 1e6).toFixed(2)} MB` : '—'}
                </td>
                <td className="num">{num(file.rows)}</td>
                {index === 0 && (
                  <td rowSpan={entry.files.length}>
                    <div className="row tight">
                      {(['qa', 'corpus'] as const).map((kind) => {
                        const open = preview?.dataset === entry.name && preview.kind === kind
                        return (
                          <button
                            key={kind}
                            className={`action small${open ? ' primary' : ''}`}
                            disabled={!entry.files_ready}
                            onClick={() =>
                              setPreview(open ? null : { dataset: entry.name, kind })
                            }
                          >
                            {kind === 'qa' ? '样本' : '语料'}
                          </button>
                        )
                      })}
                    </div>
                  </td>
                )}
              </tr>
            )),
          )}
        </tbody>
      </table>

      {preview && (
        <RawPreview
          key={`${preview.dataset}:${preview.kind}`}
          dataset={preview.dataset}
          kind={preview.kind}
        />
      )}
    </>
  )
}

function RawPreview({ dataset, kind }: { dataset: string; kind: 'qa' | 'corpus' }) {
  const [offset, setOffset] = useState(0)
  const [term, setTerm] = useState('')
  const [q, setQ] = useState('')
  const limit = 1
  const { data, error, loading } = useAsync(
    () => api.rawSamples(dataset, { kind, q, limit, offset }),
    [dataset, kind, q, offset],
  )

  if (loading) return <Loading what="原始样例" />
  if (error) return <Failed error={error} />
  if (!data) return null

  const row = data.rows[0]

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>
          {dataset} 原始{kind === 'qa' ? '样本' : '语料'}
        </h3>
        <span className="small muted mono">{data.source_file}</span>
      </div>
      <div className="record-filters" style={{ marginTop: 10 }}>
        <RecordSearch
          placeholder={`搜索原始${kind === 'qa' ? '样本' : '语料'}的任意字段`}
          value={term}
          onChange={setTerm}
          onSearch={(value) => {
            setQ(value)
            setOffset(0)
          }}
        />
      </div>
      <pre className="block tall">{row ? JSON.stringify(row, null, 2) : '（没有内容）'}</pre>
      <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} itemLabels />
    </div>
  )
}

export type { DatasetEntry }
