import { useEffect, useState } from 'react'
import { api, getToken, setToken } from '../api'
import type {
  AkashaFeature,
  AkashaModelProvider,
  Connection,
  ConnectionTest,
  ModelConfig,
  Provider,
  ProviderProbe,
} from '../types'
import { Failed, Field, Loading, Pass, SecretField, useAction, useAsync } from '../ui'

/** 配置层：Akasha 连接、它那边的模型配置、本地 judge / 归因端点。 */
export function Settings() {
  return (
    <>
      <div className="panel-head">
        <h2>配置</h2>
        <div className="row tight">
          <ImportButton />
          <ExportButton />
        </div>
      </div>
      <AccessToken />
      <ConnectionForm />
      <ModelConfigs />
      <Providers role="judge" title="评估模型 judge" />
      <Providers role="attribution" title="归因模型 attribute" />
    </>
  )
}

/** 把全部配置（含明文密钥）导出为一份 JSON 文件。 */
function ExportButton() {
  const save = useAction<unknown>()
  return (
    <button
      className="action"
      disabled={save.busy}
      onClick={() =>
        save.run(async () => {
          const data = await api.exportConfig()
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `akasha-config-${new Date().toISOString().slice(0, 10)}.json`
          a.click()
          URL.revokeObjectURL(url)
          return data
        })
      }
    >
      {save.busy ? '导出中…' : '导出配置'}
    </button>
  )
}

/** 从一份导出的 JSON 回填全部配置。导入后刷新页面让各面板重取。 */
function ImportButton() {
  const load = useAction<unknown>()

  const pick = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'application/json,.json'
    input.onchange = () => {
      const file = input.files?.[0]
      if (!file) return
      load.run(async () => {
        const text = await file.text()
        const data = JSON.parse(text) as Record<string, unknown>
        const result = await api.importConfig(data)
        window.location.reload()
        return result
      })
    }
    input.click()
  }

  return (
    <button className="action" disabled={load.busy} onClick={pick}>
      {load.busy ? '导入中…' : '导入配置'}
    </button>
  )
}

/** 访问令牌。health 返回 401 时也要显示这一段，否则没有地方改它。 */
function AccessToken() {
  const [saved, setSaved] = useState(getToken())
  const [value, setValue] = useState(saved)
  const health = useAsync(() => api.health(), [saved])
  const required = Boolean(health.data?.settings.auth_required)

  const apply = (next: string) => {
    setToken(next)
    setValue(next)
    setSaved(next)
  }

  if (!required && !saved && !health.error) return null
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>访问令牌</h3>
        {saved && !health.error && <Pass ok yes="已生效" />}
      </div>
      {health.error && <Failed error={health.error} />}

      <div className="form-grid">
        <SecretField
          label="X-Auth-Token"
          value={value}
          onChange={setValue}
          placeholder="启动时设的那个令牌"
          wide
        />
      </div>

      <div className="panel-actions">
        <button className="action primary" disabled={value === saved} onClick={() => apply(value)}>
          保存
        </button>
        <button className="action" disabled={!saved} onClick={() => apply('')}>
          清除
        </button>
        {value !== saved && <span className="small muted">未保存</span>}
      </div>
    </div>
  )
}

// 各数值字段的下限。间隔可以是 0，超时不行；模型调用并发由各运行层选择。
const NUMBER_FIELDS = [
  ['timeout_seconds', '模型请求超时（秒）', 1],
  ['request_interval_seconds', '模型请求间隔（秒）', 0],
] as const

type Form = Record<string, string>

/** 数值也按字符串存，这样清空输入框能留着空而不变成 0。 */
function toForm(data: Connection): Form {
  const { compiles: _compiles, updated_at: _updated, ...rest } = data
  return Object.fromEntries(Object.entries(rest).map(([key, value]) => [key, String(value)]))
}

function ConnectionForm() {
  const { data, error, loading, reload } = useAsync(() => api.connection(), [])
  const [form, setForm] = useState<Form>({})
  const [loaded, setLoaded] = useState<Form>({})
  const save = useAction<{ updated: string[] }>()
  const test = useAction<ConnectionTest>()

  useEffect(() => {
    if (data) {
      setForm(toForm(data))
      setLoaded(toForm(data))
    }
  }, [data])

  if (loading && !data) return <Loading what="连接配置" />
  if (error) return <Failed error={error} />
  if (!data) return null

  // 表单一改就清掉旧的保存结果与测试结论，它们不再对应当前表单。
  const set = (key: string, value: string) => {
    setForm((f) => ({ ...f, [key]: value }))
    save.reset()
    test.reset()
  }
  const revert = () => {
    setForm(loaded)
    save.reset()
    test.reset()
  }

  const dirty = Object.keys(loaded).some((key) => form[key] !== loaded[key])
  const invalid = NUMBER_FIELDS.filter(([key, , min]) => {
    const value = Number(form[key])
    return form[key]?.trim() === '' || !Number.isFinite(value) || value < min
  })

  const payload = () => ({
    ...form,
    ...Object.fromEntries(NUMBER_FIELDS.map(([key]) => [key, Number(form[key])])),
  })

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Akasha 连接</h3>
        {dirty && <span className="tag warn">有未保存的改动</span>}
      </div>

      {save.error && <Failed error={save.error} />}
      {test.error && <Failed error={test.error} />}
      {save.result && (
        <div className="note">已保存 {save.result.updated.length} 个字段。</div>
      )}

      <div className="form-grid">
        <Field label="base_url">
          <input value={form.base_url ?? ''} onChange={(e) => set('base_url', e.target.value)} />
        </Field>
        <Field label="登录邮箱">
          <input value={form.email ?? ''} onChange={(e) => set('email', e.target.value)} />
        </Field>
        <SecretField
          label="登录密码"
          value={form.password ?? ''}
          onChange={(value) => set('password', value)}
        />
        <SecretField
          label="只读 PostgreSQL"
          hint="归因链路用，可不填"
          value={form.database_url ?? ''}
          onChange={(value) => set('database_url', value)}
          placeholder="postgresql://…"
        />
      </div>

      <div className="form-grid compact">
        {NUMBER_FIELDS.map(([key, label, min]) => (
          <Field key={key} label={label}>
            <input
              type="number"
              min={min}
              value={form[key] ?? ''}
              onChange={(e) => set(key, e.target.value)}
            />
          </Field>
        ))}
      </div>

      {invalid.length > 0 && (
        <div className="note bad">
          <strong>这些字段要填数字：</strong>
          {invalid.map(([, label, min]) => `${label}（≥ ${min}）`).join('、')}
        </div>
      )}

      <div className="panel-actions">
        <button
          className="action primary"
          disabled={save.busy || !dirty || invalid.length > 0}
          onClick={() =>
            save.run(async () => {
              const result = await api.saveConnection(payload())
              reload()
              return result
            })
          }
        >
          {save.busy ? '保存中…' : '保存'}
        </button>
        <button className="action" disabled={!dirty || save.busy} onClick={revert}>
          放弃改动
        </button>
        <button
          className="action"
          disabled={test.busy}
          onClick={() => test.run(() => api.testConnection())}
        >
          {test.busy ? '测试中…' : '测试连接'}
        </button>
        {dirty && <span className="small muted">测试连接用的是已保存的配置</span>}
      </div>

      {test.result && (
        <div className={`note${test.result.is_owner ? '' : ' warn'}`}>
          <strong>连接成功。</strong> {test.result.user.email} · 角色 {test.result.user.role} ·{' '}
          workspace {test.result.workspace.name ?? test.result.workspace.id}
          <span style={{ marginLeft: 8 }}>
            <Pass ok={test.result.is_owner} yes="owner" no="非 owner" />
          </span>
          {test.result.owner_warning && (
            <div className="small" style={{ marginTop: 4 }}>
              {test.result.owner_warning}
            </div>
          )}
        </div>
      )}

      {test.result && test.result.blocked_compiles.length > 0 && (
        <div className="note bad">
          <strong>{test.result.blocked_compiles.length} 次编译在当前连接下用不了。</strong>
          <ul>
            {test.result.blocked_compiles.map((entry) => (
              <li key={entry.id} className="small">
                <span className="mono">{entry.run_id}</span> · {entry.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

    </div>
  )
}

/** 四项配置各自的标题。 */
const FEATURE_LABELS: Record<string, { title: string }> = {
  compiler: { title: '编译模型' },
  embedding: { title: '嵌入模型' },
  answer: { title: '回答模型' },
  image: { title: '图像模型' },
}

/** 远端 live 配置只读展示，编辑改在本地组里做。 */
function ModelConfigs() {
  const { data, error, loading } = useAsync(() => api.modelConfigs(), [])

  if (loading && !data) return <Loading what="模型配置" />

  return (
    <>
      <div className="panel">
        <h3>Akasha 远端模型配置</h3>
        {error ? (
          <>
            <Failed error={error} />
            <p className="small muted">这一段需要连上 Akasha。先把上面的连接配好并测试通过。</p>
          </>
        ) : (
          data && <LiveTable data={data} />
        )}
      </div>

      <AkashaModels />
    </>
  )
}

function LiveTable({ data }: { data: { features: string[]; live: unknown } }) {
  const live: ModelConfig[] = Array.isArray(data.live)
    ? data.live
    : ((data.live as { configs?: ModelConfig[] }).configs ?? [])
  const byFeature = new Map(live.map((entry) => [entry.feature, entry]))
  return (
    <table>
      <thead>
        <tr>
          <th>配置项</th>
          <th>模型</th>
          <th>base_url(/v1)</th>
          <th>密钥</th>
        </tr>
      </thead>
      <tbody>
        {data.features.map((feature) => {
          const row = byFeature.get(feature)
          const label = FEATURE_LABELS[feature] ?? { title: feature }
          return (
            <tr key={feature}>
              <td>
                <div className="stack">
                  <span>{label.title}</span>
                  <span className="mono muted">{feature}</span>
                </div>
              </td>
              <td className="small mono">{row?.model ?? '—'}</td>
              <td className="small mono muted truncate">{row?.baseUrl ?? '—'}</td>
              <td>
                <Pass ok={Boolean(row?.apiKeySet)} yes="已设置" no="缺失" />
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

const BLANK = { label: '', base_url: '', model: '', api_key: '' }

type AkashaModelForm = {
  feature: AkashaFeature
  label: string
  base_url: string
  model: string
  api_key: string
  dimension: string
  parameters: Record<string, unknown>
}

const emptyAkashaModel = (feature: AkashaFeature): AkashaModelForm => ({
  feature, label: '', base_url: '', model: '', api_key: '', dimension: '', parameters: {},
})

/** 四类 Akasha 模型分开保存，编译与查询在各自页面自由组合。 */
function AkashaModels() {
  const { data, error, loading, reload } = useAsync(() => api.akashaModels(), [])
  const [editing, setEditing] = useState<number | 'new' | null>(null)
  const [form, setForm] = useState<AkashaModelForm>(emptyAkashaModel('answer'))
  const save = useAction<unknown>()
  const remove = useAction<unknown>()
  const apply = useAction<unknown>()
  const models = data?.models ?? []

  if (loading && !data) return <Loading what="Akasha 模型配置" />

  const edit = (item: AkashaModelProvider) => {
    setEditing(item.id)
    setForm({
      feature: item.feature, label: item.label, base_url: item.base_url,
      model: item.model, api_key: '',
      dimension: item.parameters.dimension === undefined ? '' : String(item.parameters.dimension),
      parameters: item.parameters,
    })
    save.reset()
  }
  const create = (feature: AkashaFeature) => {
    setEditing('new')
    setForm(emptyAkashaModel(feature))
    save.reset()
  }
  const close = () => {
    setEditing(null)
    save.reset()
  }
  const dimension = Number(form.dimension)
  const dimensionInvalid = form.feature === 'embedding' && form.dimension.trim() !== ''
    && (!Number.isInteger(dimension) || dimension <= 0)
  const label = form.label.trim()
  const taken = models.some((item) =>
    item.feature === form.feature && item.label === label && item.id !== editing)

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Akasha 独立模型配置</h3>
      </div>
      {error && <Failed error={error} />}
      {save.error && <Failed error={save.error} />}
      {remove.error && <Failed error={remove.error} />}
      {apply.error && <Failed error={apply.error} />}
      {data?.features.map((feature) => (
        <div key={feature} className="metric-family">
          <div className="spread">
            <h4>{FEATURE_LABELS[feature]?.title ?? feature}</h4>
            <button className="action small" disabled={save.busy} onClick={() => create(feature)}>
              新增端点
            </button>
          </div>
          <table className={`endpoint-table${feature === 'embedding' ? ' embedding-endpoint-table' : ''}`}>
            <colgroup>
              <col className="endpoint-label-col" />
              <col className="endpoint-name-col" />
              <col className="endpoint-url-col" />
              {feature === 'embedding' && <col className="endpoint-dimension-col" />}
              <col className="endpoint-key-col" />
              <col className="endpoint-actions-col" />
            </colgroup>
            <thead><tr><th>标签</th><th>模型</th><th>base_url(/v1)</th>{feature === 'embedding' && <th>维度</th>}<th>密钥</th><th /></tr></thead>
            <tbody>
              {models.filter((item) => item.feature === feature).map((item) => (
                <tr key={item.id} className={item.id === editing ? 'selected' : ''}>
                  <td>{item.label}</td>
                  <td className="mono small">{item.model}</td>
                  <td className="mono small muted truncate">{item.base_url}</td>
                  {feature === 'embedding' && <td className="mono small">{String(item.parameters.dimension ?? '—')}</td>}
                  <td><Pass ok={item.api_key_set} yes="已设置" no="缺失" /></td>
                  <td className="table-actions-cell"><div className="table-actions">
                    <button className="action small" onClick={() => edit(item)}>编辑</button>
                    <button className="action small" disabled={apply.busy} onClick={() => apply.run(() => api.applyAkashaModel(item.id))}>应用</button>
                    <button className="action small danger" disabled={remove.busy} onClick={() => {
                      if (!window.confirm(`删除「${item.label}」？`)) return
                      remove.run(async () => { const result = await api.deleteAkashaModel(item.id); reload(); return result })
                    }}>删除</button>
                  </div></td>
                </tr>
              ))}
            </tbody>
          </table>
          {models.every((item) => item.feature !== feature) && editing === null && (
            <p className="small muted">还没有配端点。</p>
          )}
          {editing !== null && form.feature === feature && (
            <>
              <h4>{typeof editing === 'number' ? `编辑端点 #${editing}` : '新增端点'}</h4>
              <div className="form-grid inline">
                <Field label="标签" hint="同一类型下不重名"><input value={form.label} onChange={(e) => setForm({ ...form, label: e.target.value })} placeholder="default" /></Field>
                <Field label="模型"><input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} /></Field>
                <Field label="base_url"><input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} placeholder="https://api.example.com/v1" /></Field>
                <SecretField label="api_key" hint={typeof editing === 'number' ? '留空保留原值' : undefined} value={form.api_key} onChange={(api_key) => setForm({ ...form, api_key })} />
                {feature === 'embedding' && (
                  <Field label="向量维度" hint="可选，填写正整数">
                    <input type="number" min={1} step={1} value={form.dimension} onChange={(e) => setForm({ ...form, dimension: e.target.value })} placeholder="1024" />
                  </Field>
                )}
              </div>
              {dimensionInvalid && <div className="note bad">向量维度必须是正整数。</div>}
              {taken && <div className="note bad">该类型下已经有一个叫「{label}」的端点。</div>}
              <div className="panel-actions">
                <button className="action primary" disabled={save.busy || dimensionInvalid || taken || !label || !form.model.trim() || !form.base_url.trim()} onClick={() => save.run(async () => {
                  const parameters = { ...form.parameters }
                  if (feature === 'embedding') {
                    if (form.dimension.trim()) parameters.dimension = dimension
                    else delete parameters.dimension
                  }
                  const result = await api.saveAkashaModel({
                    feature: form.feature, label, base_url: form.base_url.trim(),
                    model: form.model.trim(), api_key: form.api_key, parameters,
                    ...(typeof editing === 'number' ? { id: editing } : {}),
                  })
                  close(); reload(); return result
                })}>保存</button>
                <button className="action" onClick={close}>取消</button>
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  )
}

/** 一行端点。探测与删除的状态逐行独立，所以拆成组件。 */
function ProviderRow({
  provider,
  selected,
  onEdit,
  onDeleted,
}: {
  provider: Provider
  selected: boolean
  onEdit: () => void
  onDeleted: () => void
}) {
  const probe = useAction<ProviderProbe>()
  const remove = useAction<unknown>()

  return (
    <>
      <tr className={selected ? 'selected' : ''}>
        <td>{provider.label}</td>
        <td className="small mono">{provider.model}</td>
        <td className="small mono muted truncate">{provider.base_url}</td>
        <td>
          <Pass ok={provider.api_key_set} yes="已设置" no="缺失" />
        </td>
        <td>
          <div className="row tight">
            <button className="action small" onClick={onEdit}>
              编辑
            </button>
            <button
              className="action small"
              disabled={probe.busy}
              title="向这个端点发一句 hi，真实调用模型"
              onClick={() => probe.run(() => api.probeProvider(provider.id))}
            >
              {probe.busy ? '探测中…' : '探测'}
            </button>
            <button
              className="action small danger"
              disabled={remove.busy}
              onClick={() => {
                if (
                  !window.confirm(
                    `删除端点「${provider.label}」？\n\n引用它的评测与归因记录会失去关联。`,
                  )
                )
                  return
                remove.run(async () => {
                  const result = await api.deleteProvider(provider.id)
                  onDeleted()
                  return result
                })
              }}
            >
              删除
            </button>
          </div>
        </td>
      </tr>
      {(probe.result || probe.error || remove.error) && (
        <tr>
          <td colSpan={5}>
            {remove.error && <Failed error={remove.error} />}
            {probe.error && <Failed error={probe.error} />}
            {probe.result && <ProbeResult result={probe.result} onClose={probe.reset} />}
          </td>
        </tr>
      )}
    </>
  )
}

/** 探测结果。失败时摊开 provider 回的原文。 */
function ProbeResult({ result, onClose }: { result: ProviderProbe; onClose: () => void }) {
  return (
    <div className={`note ${result.ok ? 'ok' : 'bad'}`}>
      <div className="spread">
        <strong>
          {result.ok ? 'Success' : 'Fail'}
          {result.status !== null && <span className="small mono muted"> HTTP {result.status}</span>}
          {result.failure && <span className="small mono"> {result.failure}</span>}
        </strong>
        <button className="action small" onClick={onClose}>
          收起
        </button>
      </div>
      {result.ok && result.reply && (
        <pre className="block" style={{ marginTop: 6 }}>
          {result.reply}
        </pre>
      )}
      {!result.ok && result.detail && (
        <pre className="block" style={{ marginTop: 6 }}>
          {result.detail}
        </pre>
      )}
    </div>
  )
}

/** 表单默认收起，由「新增端点」或表格里的「编辑」打开。
 *
 * 编辑时提交 id：后端按 (role, label) upsert，不带 id 会把改名变成新增。
 */
function Providers({ role, title }: { role: 'judge' | 'attribution'; title: string }) {
  const { data, error, loading, reload } = useAsync<Provider[]>(() => api.providers(role), [role])
  // null 是收起来，'new' 是新建，数字是在改那一条。
  const [mode, setMode] = useState<number | 'new' | null>(null)
  const [form, setForm] = useState(BLANK)
  const save = useAction<{ id: number }>()

  const providers = data ?? []
  const editing = typeof mode === 'number' ? mode : null
  const set = (patch: Partial<typeof form>) => {
    setForm({ ...form, ...patch })
    save.reset()
  }
  const close = () => {
    setMode(null)
    setForm(BLANK)
    save.reset()
  }
  const create = () => {
    setMode('new')
    setForm(BLANK)
    save.reset()
  }
  const edit = (provider: Provider) => {
    setMode(provider.id)
    setForm({
      label: provider.label,
      base_url: provider.base_url,
      model: provider.model,
      api_key: '',
    })
    save.reset()
  }

  // 新建时标签不能撞已有的，否则会覆盖那一条。
  const label = form.label.trim() || 'default'
  const taken = providers.some((p) => p.label === label && p.id !== editing)

  if (loading && !data) return <Loading what={title} />

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <div className="row tight">
          <span className="small muted">{providers.length} 个端点</span>
          <button className="action small" disabled={save.busy} onClick={create}>
            新增端点
          </button>
        </div>
      </div>

      {error && <Failed error={error} />}
      {save.error && <Failed error={save.error} />}

      {providers.length > 0 && (
        <table className="endpoint-table">
          <colgroup>
            <col className="endpoint-label-col" />
            <col className="endpoint-name-col" />
            <col className="endpoint-url-col" />
            <col className="endpoint-key-col" />
            <col className="endpoint-actions-col" />
          </colgroup>
          <thead>
            <tr>
              <th>标签</th>
              <th>模型</th>
              <th>base_url(/v1)</th>
              <th>密钥</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {providers.map((provider) => (
              <ProviderRow
                key={provider.id}
                provider={provider}
                selected={provider.id === editing}
                onEdit={() => edit(provider)}
                onDeleted={() => {
                  if (editing === provider.id) close()
                  reload()
                }}
              />
            ))}
          </tbody>
        </table>
      )}

      {providers.length === 0 && !error && mode === null && (
        <p className="small muted">还没有配端点。</p>
      )}

      {save.result && <div className="note">已保存端点。</div>}

      {mode !== null && (
        <>
          <h4>{editing === null ? '新增端点' : `编辑端点 #${editing}`}</h4>
          <div className="form-grid inline">
            <Field label="标签" hint="同一类型下不重名">
              <input
                value={form.label}
                onChange={(e) => set({ label: e.target.value })}
                placeholder="default"
              />
            </Field>
            <Field label="模型">
              <input value={form.model} onChange={(e) => set({ model: e.target.value })} />
            </Field>
            <Field label="base_url">
              <input
                value={form.base_url}
                onChange={(e) => set({ base_url: e.target.value })}
                placeholder="https://api.example.com/v1"
              />
            </Field>
            <SecretField
              label="api_key"
              hint={editing === null ? undefined : '留空保留原值'}
              value={form.api_key}
              onChange={(value) => set({ api_key: value })}
            />
          </div>

          {taken && (
            <div className="note bad">
              已经有一个叫「{label}」的端点了。换个标签，或者在上表里点它的「编辑」。
            </div>
          )}
        </>
      )}

      {mode !== null && (
        <div className="panel-actions">
          <>
            <button
              className="action primary"
              disabled={save.busy || taken || !form.base_url || !form.model}
              onClick={() =>
                save.run(async () => {
                  const result = await api.saveProvider(role, {
                    ...form,
                    label,
                    ...(editing === null ? {} : { id: editing }),
                  })
                  setMode(null)
                  setForm(BLANK)
                  reload()
                  return result
                })
              }
            >
              {save.busy ? '保存中…' : '保存'}
            </button>
            <button className="action" disabled={save.busy} onClick={close}>
              取消
            </button>
          </>
        </div>
      )}
    </div>
  )
}
