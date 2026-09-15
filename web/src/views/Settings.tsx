import { useEffect, useState } from 'react'
import { api, getToken, setToken } from '../api'
import type {
  AkashaConfigGroup,
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

// 各数值字段的下限。间隔可以是 0，超时不行。并发挪到各模型端点。
const NUMBER_FIELDS = [
  ['timeout_seconds', '请求超时（秒）', 1],
  ['request_interval_seconds', '请求间隔（秒）', 0],
  ['poll_interval_seconds', '编译轮询间隔（秒）', 0],
  ['poll_timeout_seconds', '编译轮询超时（秒）', 1],
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

      {test.result?.group_drift &&
        Object.values(test.result.group_drift.drift).some(Boolean) && (
          <div className="note warn">
            本地选中组「{test.result.group_drift.label}」与远端配置不一致（
            {Object.entries(test.result.group_drift.drift)
              .filter(([, changed]) => changed)
              .map(([feature]) => feature)
              .join('、')}
            ）。可在下方「应用到 Akasha」推送整组。
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

      <AkashaGroups />
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

type GroupForm ={ label: string; configs: Record<string, { model: string; baseUrl: string; apiKey: string }> }

const blankGroupForm = (features: string[]): GroupForm => ({
  label: '',
  configs: Object.fromEntries(features.map((f) => [f, { model: '', baseUrl: '', apiKey: '' }])),
})

/** 本地保存的 Akasha 模型配置组：多组可存，选中一组可整组应用到远端。 */
function AkashaGroups() {
  const { data, error, loading, reload } = useAsync(() => api.akashaConfigs(), [])
  // null 收起，'new' 新建，数字是在改那一条。
  const [mode, setMode] = useState<number | 'new' | null>(null)
  const [form, setForm] = useState<GroupForm>({ label: '', configs: {} })
  const save = useAction<{ id: number }>()

  const features = data?.features ?? []
  const groups = data?.groups ?? []
  const editing = typeof mode === 'number' ? mode : null

  const close = () => {
    setMode(null)
    save.reset()
  }
  const create = () => {
    setForm(blankGroupForm(features))
    setMode('new')
    save.reset()
  }
  const edit = (group: AkashaConfigGroup) => {
    setForm({
      label: group.label,
      configs: Object.fromEntries(
        features.map((f) => [
          f,
          {
            model: group.configs[f]?.model ?? '',
            baseUrl: group.configs[f]?.baseUrl ?? '',
            apiKey: '',
          },
        ]),
      ),
    })
    setMode(group.id)
    save.reset()
  }

  const label = form.label.trim()
  const taken = groups.some((g) => g.label === label && g.id !== editing)

  const setFeature = (
    feature: string,
    patch: Partial<{ model: string; baseUrl: string; apiKey: string }>,
  ) => {
    const current = form.configs[feature] ?? { model: '', baseUrl: '', apiKey: '' }
    setForm({ ...form, configs: { ...form.configs, [feature]: { ...current, ...patch } } })
    save.reset()
  }

  if (loading && !data) return <Loading what="配置组" />

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Akasha 本地配置组</h3>
        <span className="small muted">{groups.length} 组</span>
      </div>

      {error && <Failed error={error} />}
      {save.error && <Failed error={save.error} />}

      {groups.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>组</th>
              {features.map((f) => (
                <th key={f}>{f}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => (
              <GroupRow
                key={group.id}
                group={group}
                features={features}
                selected={group.id === editing}
                onEdit={() => edit(group)}
                onChanged={() => {
                  if (editing === group.id) close()
                  reload()
                }}
              />
            ))}
          </tbody>
        </table>
      )}

      {groups.length === 0 && !error && mode === null && (
        <p className="small muted">还没有配置组。</p>
      )}

      {mode !== null && (
        <>
          <h4>{editing === null ? '新建配置组' : `编辑配置组 #${editing}`}</h4>
          <Field label="组名" hint="不重名">
            <input
              value={form.label}
              onChange={(e) => {
                setForm({ ...form, label: e.target.value })
                save.reset()
              }}
              placeholder="例如：gpt-4o 一组"
            />
          </Field>
          {features.map((feature) => (
            <div key={feature} className="form-grid inline" style={{ marginTop: 8 }}>
              <Field label={`${feature} 模型`}>
                <input
                  value={form.configs[feature]?.model ?? ''}
                  onChange={(e) => setFeature(feature, { model: e.target.value })}
                />
              </Field>
              <Field label="base_url">
                <input
                  value={form.configs[feature]?.baseUrl ?? ''}
                  onChange={(e) => setFeature(feature, { baseUrl: e.target.value })}
                  placeholder="https://api.example.com/v1"
                />
              </Field>
              <SecretField
                label="api_key"
                hint={editing === null ? undefined : '留空保留原值'}
                value={form.configs[feature]?.apiKey ?? ''}
                onChange={(value) => setFeature(feature, { apiKey: value })}
              />
            </div>
          ))}

          {taken && (
            <div className="note bad">已经有一个叫「{label}」的组了。换个名字。</div>
          )}
        </>
      )}

      <div className="panel-actions">
        {mode === null ? (
          <button className="action" onClick={create}>
            新建配置组
          </button>
        ) : (
          <>
            <button
              className="action primary"
              disabled={save.busy || taken || !label}
              onClick={() =>
                save.run(async () => {
                  const result = await api.saveAkashaConfig({
                    label,
                    configs: form.configs,
                    ...(editing === null ? {} : { id: editing }),
                  })
                  close()
                  reload()
                  return result
                })
              }
            >
              {save.busy ? '保存中…' : editing === null ? '创建组' : '保存修改'}
            </button>
            <button className="action" disabled={save.busy} onClick={close}>
              取消
            </button>
          </>
        )}
      </div>
    </div>
  )
}

/** 一行配置组。选中、应用、删除的状态逐行独立。 */
function GroupRow({
  group,
  features,
  selected,
  onEdit,
  onChanged,
}: {
  group: AkashaConfigGroup
  features: string[]
  selected: boolean
  onEdit: () => void
  onChanged: () => void
}) {
  const apply = useAction<{ applied: string[]; impact: string; requires_new_compile: boolean }>()
  const remove = useAction<unknown>()

  return (
    <>
      <tr className={selected ? 'selected' : ''}>
        <td>
          <div className="stack">
            <span>
              {group.label}
              {group.selected && (
                <span className="tag" style={{ marginLeft: 6 }}>
                  选中
                </span>
              )}
            </span>
          </div>
        </td>
        {features.map((f) => (
          <td key={f} className="small mono truncate">
            {group.configs[f]?.model || '—'}
          </td>
        ))}
        <td>
          <div className="row tight">
            <button className="action small" onClick={onEdit}>
              编辑
            </button>
            <button
              className="action small primary"
              disabled={apply.busy}
              title="把这一组四项配置整组推送到远端 Akasha，并置为选中"
              onClick={() => {
                if (
                  !window.confirm(
                    `把「${group.label}」整组应用到远端 Akasha？\n\ncompiler / embedding 若变会导致已有编译不可比。`,
                  )
                )
                  return
                apply.run(async () => {
                  const result = await api.applyAkashaConfig(group.id)
                  onChanged()
                  return result
                })
              }}
            >
              {apply.busy ? '应用中…' : '应用'}
            </button>
            <button
              className="action small danger"
              disabled={remove.busy}
              onClick={() => {
                if (!window.confirm(`删除配置组「${group.label}」？`)) return
                remove.run(async () => {
                  const result = await api.deleteAkashaConfig(group.id)
                  onChanged()
                  return result
                })
              }}
            >
              删除
            </button>
          </div>
        </td>
      </tr>
      {(apply.result || apply.error || remove.error) && (
        <tr>
          <td colSpan={features.length + 2}>
            {apply.error && <Failed error={apply.error} />}
            {remove.error && <Failed error={remove.error} />}
            {apply.result && (
              <div className={`note${apply.result.requires_new_compile ? ' warn' : ''}`}>
                已应用 {apply.result.applied?.length ?? 4} 项。{apply.result.impact}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  )
}

const BLANK = { label: '', base_url: '', model: '', api_key: '', concurrency: 1 }

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
        <td className="small mono">{provider.concurrency}</td>
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
          <td colSpan={6}>
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

/** 表单默认收起，由「新建端点」或表格里的「编辑」打开。
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
      concurrency: provider.concurrency,
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
        <span className="small muted">{providers.length} 个端点</span>
      </div>

      {error && <Failed error={error} />}
      {save.error && <Failed error={save.error} />}

      {providers.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>标签</th>
              <th>模型</th>
              <th>base_url(/v1)</th>
              <th>密钥</th>
              <th>并发</th>
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
          <h4>{editing === null ? '新建端点' : `编辑端点 #${editing}`}</h4>
          <div className="form-grid inline">
            <Field label="标签" hint="同一角色下不重名">
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
            <Field label="并发" hint="judge / 归因调用的并发数">
              <input
                type="number"
                min={1}
                max={16}
                value={form.concurrency}
                onChange={(e) => set({ concurrency: Number(e.target.value) })}
              />
            </Field>
          </div>

          {taken && (
            <div className="note bad">
              已经有一个叫「{label}」的端点了。换个标签，或者在上表里点它的「编辑」。
            </div>
          )}
        </>
      )}

      <div className="panel-actions">
        {mode === null ? (
          <button className="action" onClick={create}>
            新建端点
          </button>
        ) : (
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
              {save.busy ? '保存中…' : editing === null ? '创建端点' : '保存修改'}
            </button>
            <button className="action" disabled={save.busy} onClick={close}>
              取消
            </button>
          </>
        )}
      </div>
    </div>
  )
}
