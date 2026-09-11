import { api } from '../api'
import type { Diff, Lineage } from '../types'
import { Collapsible, Loading, num, percent, useAsync } from '../ui'

/** 原文 vs 编译产物的并排 diff。**一等视图，不是附属功能。**
 *
 * 理由具体：那次 recall@5 = 0.5 的根因只有把两者并排才看得见 —— 原文里有
 * "Grammy and Emmy award winning"，编译产物写成了 "guest artists including",
 * 而问题问的正是 "who won Grammy and Emmy award"。于是三条召回路径同时断：
 * 词法（词不在索引文本里）、稠密（主题漂了）、图扩展（那条边不存在）。
 *
 * 编译**不是压缩而是扩写**（实测中位 2.19 倍，仅 2.7% 净压缩），所以丢修饰语
 * 是改写策略而非空间不足 —— 那个比值让读者自己看到这一点。
 */
export function DocDiff({
  pageId,
  question = '',
  uploaded,
}: {
  pageId: string
  question?: string
  uploaded?: string
}) {
  const { data, error, loading } = useAsync(
    () => api.diff(pageId, question),
    [pageId, question],
  )

  if (loading) return <Loading what="编译 diff" />
  if (error)
    return (
      <div className="note warn">
        <strong>看不到编译产物。</strong> {error}
        <div className="small" style={{ marginTop: 4 }}>
          原文/编译 diff 需要 Akasha 的只读数据库连接（database_url），
          到「配置」层填上。其余视图不需要它。
        </div>
        {uploaded && (
          <>
            <h4>导入 Akasha 的正文</h4>
            <pre className="block">{uploaded}</pre>
          </>
        )}
      </div>
    )
  if (!data) return null

  return <DiffPanels diff={data} pageId={pageId} />
}

export function DiffPanels({ diff, pageId }: { diff: Diff; pageId?: string }) {
  const lost = diff.question_terms_lost
  const empty = !diff.compiled.text && !diff.source.text

  return (
    <>
      {lost.length > 0 ? (
        <div className="note bad">
          <strong>问题里有 {lost.length} 个实词不在编译产物里。</strong>
          <div style={{ marginTop: 6 }}>
            {lost.map((term) => (
              <span key={term} className="chip lost">
                {term}
              </span>
            ))}
          </div>
          <div className="small" style={{ marginTop: 6 }}>
            {diff.verdict} —— 这不是调参能救的，词已经不在被索引的文本里。
          </div>
        </div>
      ) : (
        <div className="note plain small">{diff.verdict}</div>
      )}

      <div className="row small muted">
        <span>
          扩写比{' '}
          <strong className="mono">{num(diff.diff.expansion_ratio, 2)}×</strong>
        </span>
        <span>
          实词保留率 <strong className="mono">{percent(diff.diff.retention)}</strong>
        </span>
        <span>
          原文 {diff.diff.source_chars} 字 / 编译 {diff.diff.compiled_chars} 字
        </span>
        <span>
          丢词 {diff.diff.dropped_total} / 新增 {diff.diff.added_total}
        </span>
      </div>

      <div className="side-by-side" style={{ marginTop: 10 }}>
        <div>
          <h4>原文（不参与召回）</h4>
          <p className="small muted">
            存在 <code>knowledge_source_chunks</code>，只在解析引用时提供证据窗口。
            {diff.source.chunk_count} 个 chunk。
          </p>
          <pre className="block tall">{diff.source.text || '（空）'}</pre>
        </div>
        <div>
          <h4>编译产物（这才是被检索的文本）</h4>
          <p className="small muted">
            存在 <code>knowledge_chunks</code>，向量与词法召回都跑在它上面。
            {diff.compiled.artifact_count} 个 artifact / {diff.compiled.chunk_count} 个 chunk。
          </p>
          <pre className="block tall">{diff.compiled.text || '（空）'}</pre>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 10 }}>
        <div>
          <h4>编译丢掉的实词（{diff.diff.dropped_total}）</h4>
          <div>
            {diff.diff.dropped.length === 0 && <span className="small muted">无</span>}
            {diff.diff.dropped.map((term) => (
              <span key={term} className={`chip ${lost.includes(term) ? 'lost' : 'dropped'}`}>
                {term}
              </span>
            ))}
            {diff.diff.dropped_total > diff.diff.dropped.length && (
              <span className="small muted">
                {' '}
                …还有 {diff.diff.dropped_total - diff.diff.dropped.length} 个
              </span>
            )}
          </div>
        </div>
        <div>
          <h4>编译新增的实词（{diff.diff.added_total}）</h4>
          <div>
            {diff.diff.added.length === 0 && <span className="small muted">无</span>}
            {diff.diff.added.map((term) => (
              <span key={term} className="chip added">
                {term}
              </span>
            ))}
            {diff.diff.added_total > diff.diff.added.length && (
              <span className="small muted">
                {' '}
                …还有 {diff.diff.added_total - diff.diff.added.length} 个
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 空结果与「配置指向了别处」长得一样，所以要说清这次查的是哪个库。 */}
      {empty && diff.queried && (
        <div className="note warn small">
          这一页在 <span className="mono">{diff.queried.base_url}</span> 那个库里没有编译产物
          也没有原文块。如果这一层是在别的部署上入库的，那就是查错了地方 ——
          编译层的层卡片上有它入库时的 base_url。
        </div>
      )}

      {pageId && <LineageDetail pageId={pageId} />}
    </>
  )
}

/** 六跳链路的明细：artifact -> chunk -> 图边。默认折叠，展开看细节。 */
function LineageDetail({ pageId }: { pageId: string }) {
  const { data, error, loading } = useAsync(() => api.lineage(pageId), [pageId])

  if (loading || error || !data) return null

  return (
    <Collapsible
      title={
        <>
          编译链路：{data.counts.artifacts} artifact · {data.counts.chunks} chunk ·{' '}
          {data.counts.edges} 图边 · {data.counts.source_chunks} 原文块
        </>
      }
    >
      <LineageTables lineage={data} />
    </Collapsible>
  )
}

export function LineageTables({ lineage }: { lineage: Lineage }) {
  return (
    <>
      <h4>编译产物（artifact）</h4>
      <table>
        <thead>
          <tr>
            <th>标题</th>
            <th>类型</th>
            <th>合并键</th>
            <th>过期</th>
          </tr>
        </thead>
        <tbody>
          {lineage.artifacts.map((artifact) => (
            <tr key={artifact.id}>
              <td className="small">{artifact.title}</td>
              <td className="small">
                <span className="tag">{artifact.page_type}</span>
              </td>
              <td className="small mono muted truncate">{artifact.canonical_key ?? '—'}</td>
              <td className="small">
                {artifact.stale_at ? <span className="tag bad">已过期</span> : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>参与召回的 chunk</h4>
      <table>
        <thead>
          <tr>
            <th>角色</th>
            <th>通道</th>
            <th>embedding profile</th>
            <th>正文</th>
          </tr>
        </thead>
        <tbody>
          {lineage.chunks.map((chunk) => (
            <tr key={chunk.id}>
              <td className="small">{chunk.chunk_role ?? '—'}</td>
              <td className="small">{chunk.retrieval_channel ?? '—'}</td>
              <td
                className="small mono muted"
                title="换 embedding 后旧 chunk 的 profile 对不上，那些 chunk 永远召回不到"
              >
                {chunk.embedding_profile ?? '—'}
              </td>
              <td className="small truncate">{chunk.text}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>图边（{lineage.edges.length}）</h4>
      {lineage.edges.length === 0 ? (
        <p className="small muted">
          没有图边。图边极稀疏，跨文档实体没连起来时多跳问题就缺一跳。
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>关系</th>
              <th>从</th>
              <th>到</th>
            </tr>
          </thead>
          <tbody>
            {lineage.edges.map((edge) => (
              <tr key={edge.id}>
                <td className="small mono">{edge.relation}</td>
                <td className="small truncate">{edge.from_title}</td>
                <td className="small truncate">{edge.to_title}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="small muted">
        relation 是自由生成的（实测 555 条边散在 377 种取值上），所以图遍历无法
        按关系类型做 —— 这里只如实列出。
      </p>
    </>
  )
}
