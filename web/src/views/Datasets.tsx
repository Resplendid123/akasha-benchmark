import { useState } from 'react'
import { api } from '../api'
import { Empty, Failed, Loading, Pager, useAsync } from '../ui'

/** 数据集层：四组的规模与标注，以及**原始样例**。
 *
 * 原始样例直接读 `dataset/*.json`，不经适配器 —— 这一栏要回答的是「上游给的
 * 是什么」，而适配器的产物已经是解释过一轮的结果。两者并排才看得出归一化
 * 做了什么，所以归一化后的形态在下一层看。
 */
export function Datasets() {
  const { data, error, loading } = useAsync(() => api.datasets(), [])
  const [open, setOpen] = useState<string | null>(null)

  if (loading) return <Loading what="数据集" />
  if (error) return <Failed error={error} />
  if (!data?.length)
    return (
      <>
        <h2>数据集</h2>
        <Empty>
          库里还没有归一化过的数据集。到「归一化」那一层看适配器状态，或从「任务」
          起一次 normalize。
        </Empty>
      </>
    )

  return (
    <>
      <h2>数据集</h2>
      <p className="lede">
        数据集声明自己<strong>拥有</strong>哪些标注，指标声明自己<strong>需要</strong>哪些，
        闸门做集合比对。所以能算哪些指标由标注决定，不由数据集名字决定 ——
        勾选范围在评测层里按这个算。
      </p>

      <table>
        <thead>
          <tr>
            <th>数据集</th>
            <th>拥有的标注</th>
            <th className="num">QA</th>
            <th className="num">语料</th>
            <th className="num">唯一问题文本</th>
            <th>gold 篇数分布（去重后）</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.map((entry) => (
            <tr key={entry.name} className={open === entry.name ? 'selected' : ''}>
              <td>{entry.name}</td>
              <td>
                {entry.provides.map((dependency) => (
                  <span key={dependency} className="tag ok">
                    {dependency}
                  </span>
                ))}
                {!entry.provides.includes('gold_docs') && (
                  <span className="tag warn" title="检索、引用、多跳指标一律省略，不报 0">
                    无 gold → 省略检索族
                  </span>
                )}
              </td>
              <td className="num">{entry.qa_rows}</td>
              <td className="num">{entry.corpus_rows}</td>
              <td className="num" title="审计归因按 sha256(query) join，重复问题那一行没法连">
                {entry.unique_question_texts}
              </td>
              <td className="small mono muted">
                {Object.entries(entry.gold_count_distribution)
                  .map(([count, n]) => `${count}篇×${n}`)
                  .join('  ')}
              </td>
              <td>
                <button
                  className="action small"
                  onClick={() => setOpen(open === entry.name ? null : entry.name)}
                >
                  {open === entry.name ? '收起' : '原始样例'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {open && <RawSamples dataset={open} />}
    </>
  )
}

/** 原始样例查看器。整行原样给出，不裁字段 —— 裁了就看不到上游还有哪些字段没用上。 */
function RawSamples({ dataset }: { dataset: string }) {
  const [offset, setOffset] = useState(0)
  const limit = 5
  const { data, error, loading } = useAsync(
    () => api.rawSamples(dataset, { limit, offset }),
    [dataset, offset],
  )

  return (
    <div className="panel" style={{ marginTop: 14 }}>
      <div className="spread">
        <h3 style={{ margin: 0 }}>{dataset} 的原始样例</h3>
        {data && <span className="small muted mono">{data.source_file}</span>}
      </div>
      {loading && <Loading what="样例" />}
      {error && <Failed error={error} />}
      {data && (
        <>
          <p className="small muted">{data.note}</p>
          {data.rows.map((row, index) => (
            <pre key={offset + index} className="block" style={{ marginBottom: 8 }}>
              {JSON.stringify(row, null, 2)}
            </pre>
          ))}
          <Pager total={data.total} offset={offset} limit={limit} onChange={setOffset} />
        </>
      )}
    </div>
  )
}
