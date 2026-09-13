// 后端返回的形状。
//
// 一条贯穿全局的建模要求（决策 19）：**指标字段「可能不存在」必须进类型**。
// narrativeqa 没有 gold 文档，整族检索指标都不存在 —— 不是 0，是没有。
// 所以指标一律走 `Metrics`（索引签名 + undefined），取值处必须显式处理缺失，
// 而不是拿到一个 0 就当分数画上去。

/** 指标名 -> 值。**缺失是有意义的状态**，不要用 0 兜底。 */
export type Metrics = Record<string, number | undefined>

export type DataDependency = 'gold_docs' | 'reference_answers'

export interface MetricDefinition {
  name: string
  family: 'retrieval' | 'qa' | 'attribution' | 'multihop' | 'judge'
  /** 这个指标需要哪些标注。空数组 = 对任何数据集都能算（judge 类）。 */
  requires: DataDependency[]
  kind: 'deterministic' | 'judge'
  /** truncation_loss 之类是越低越好，排序方向要跟着它走。 */
  higher_is_better: boolean
  per_k: boolean
  description: string
}

/** 勾选范围。交集是默认，并集里多出来的项只对部分数据集成立。 */
export interface AvailableMetrics {
  datasets: string[]
  provides: Record<string, DataDependency[]>
  per_dataset: Record<string, string[]>
  computable_for_all: string[]
  computable_for_some: string[]
  partial: string[]
  note: string
}

// --- 配置层 ---

/** Akasha 连接。**只有一份**，只能改，不能新增或删除。 */
export interface AppConnection {
  id: number
  base_url: string
  api_prefix: string
  email: string
  timeout_seconds: number
  concurrency: number
  request_interval_seconds: number
  poll_interval_seconds: number
  poll_timeout_seconds: number
  updated_at: string
  /** 库中明文存储，UI 直接读写，与 base_url / email 同路。 */
  password: string
  database_url: string
  last_checked_at: string | null
  last_check_ok: number | null
  last_check_role: string | null
  last_model_configs: unknown
  /** 已入库的层。改 base_url / email 前该看一眼 —— 可能落到另一个 workspace。 */
  ingested_layers: Array<{
    id: number
    label: string
    workspace_id: string | null
    ingested_at: string
  }>
  [key: string]: unknown
}

export interface ConnectionTest {
  ok: boolean
  user: { id: string | null; email: string | null; role: string | null }
  workspace: { id: string | null; name: string | null }
  is_owner: boolean
  /** 非 OWNER 会在授权闸门静默丢弃 chunk，症状看起来像召回质量差。 */
  owner_warning: string | null
  /** 绑在这个连接上、但入库时落在别的 workspace 的层。它们跑不了。
   *  这是一次预检 —— 同一道判据 ingest / query 在登录后也会走。 */
  blocked_layers: Array<{ id: number; label: string; reason: string }>
}

/** 这一层**入库时**跑在什么上。取自 ingest 写入的快照 —— 配置改过之后它仍然
 *  是历史真相，而 workspace 闸门比对的就是它。 */
export interface IngestIdentity {
  ingested_at: string | null
  base_url: string | null
  email: string | null
  workspace_id: string | null
  workspace_name: string | null
  akasha_user_role: string | null
}

export interface ModelConfigsView {
  features: string[]
  live: unknown
  index_layers: Array<{
    id: number
    label: string
    ingested_at: string | null
    embedding_matches: boolean
    compiler_matches: boolean
  }>
}

export interface Provider {
  id: number
  role: 'judge' | 'analysis'
  label: string
  base_url: string
  model: string
  api_key_set: boolean
  params: { temperature?: number; max_tokens?: number }
  updated_at: string
}

// --- 数据集层与归一化层 ---

export interface Dataset {
  name: string
  adapter: string
  provides: DataDependency[]
  qa_rows: number
  corpus_rows: number
  gold_count_distribution: Record<string, number>
  unique_question_texts: number
  dedup_stats: Record<string, unknown>
  identity_rules: Record<string, string>
  normalized_at: string
}

export interface AdapterStatus {
  name: string
  aliases: string[]
  adapter: string
  adapter_version: string
  implemented: boolean
  provides: DataDependency[]
  files_present: boolean
  qa_file: string
  corpus_file: string
  expected_qa_rows: number | null
  normalized: boolean
  normalized_at: string | null
  qa_rows: number | null
  corpus_rows: number | null
  blocked_reason: string | null
}

export interface AdapterList {
  adapters: AdapterStatus[]
  /** dataset/ 里没有适配器认领的文件。它们**无法入库**。 */
  unclaimed_files: string[]
  dataset_dir: string
  note: string
}

export interface RawSamples {
  dataset: string
  source_file: string
  total: number
  offset: number
  limit: number
  rows: Array<Record<string, unknown>>
  note: string
}

export interface NormalizedSample {
  dataset: string
  sample_id: string
  dataset_sample_id: string
  question: string
  answers: string[]
  gold_doc_ids: string[]
  metadata: Record<string, unknown>
}

export interface NormalizedSamples {
  dataset: string
  adapter: string
  adapter_version: string
  provides: DataDependency[]
  identity_rules: Record<string, string>
  total: number
  offset: number
  limit: number
  samples: NormalizedSample[]
}

export interface NormalizedSampleDetail extends NormalizedSample {
  gold_docs: Array<{
    doc_id: string
    title: string
    text: string
    text_truncated: boolean
  }>
  /** gold 指向语料里不存在的 doc_id —— 身份规则出错的信号。 */
  missing_gold_doc_ids: string[]
}

export interface CorpusDocs {
  dataset: string
  total: number
  offset: number
  limit: number
  docs: Array<{
    doc_id: string
    title: string
    text: string
    text_truncated: boolean
    text_sha256: string
  }>
}

// --- 编译层（库里叫索引层）---

export interface IndexLayerDataset {
  dataset: string
  space_id: string | null
  space_slug: string | null
  strategy: string
  qa_count: number
  corpus_count: number
  gold_doc_count: number
  negative_doc_count: number
}

export interface EvalLayerSummary {
  run_id: string
  id: number
  label: string
  config_hash: string
  created_at: string
  finished_at: string | null
}

export interface QueryLayerSummary {
  run_id: string
  selection: Record<string, string[]> | null
  id: number
  label: string
  config_hash: string
  score_threshold: number | null
  concurrency: number
  model_configs_match_index: number | null
  started_at: string
  finished_at: string | null
  stats: Record<string, { responses: number; failures: number; latency_mean: number | null }>
  /** 从响应行的 min/max 现算，覆盖累积的全部会话。 */
  request_window: [string, string] | null
  eval_layers: EvalLayerSummary[]
}

export interface IndexLayer {
  run_id: string
  id: number
  label: string
  /** 抽样配置的哈希。答「是同一个子集吗」，离线可算。 */
  subset_hash: string
  /** 抽样 + compiler + embedding。**入库前是 null** —— 那时身份还不完整。 */
  config_hash: string | null
  seed: number
  qa_limit: number
  negatives_ratio: number
  quality_passed: number | null
  created_at: string
  ingested_at: string | null
  datasets: IndexLayerDataset[]
  page_map_counts: Record<string, number>
  /** 三项前置闸门的合并结果：质量、导入完整性、编译是否超时。 */
  ready_for_query: boolean
  not_ready_reasons: string[]
  /** 同抽样、不同 embedding 的层。对照实验靠它找同伴。 */
  same_subset_layers: number[]
  /** 这一层入库时跑在什么上。 */
  ingest_identity: IngestIdentity
  query_layers: QueryLayerSummary[]
}

export interface LayerDoc {
  dataset: string
  doc_id: string
  is_gold: boolean
  md_sha256: string
  /** 走血缘链路的钥匙。null = 没导入成功。 */
  page_id: string | null
}

export interface LayerDocs {
  index_layer_id: number
  total: number
  imported: number
  offset: number
  limit: number
  docs: LayerDoc[]
}

export interface LayerDocDetail {
  index_layer_id: number
  dataset: string
  doc_id: string
  is_gold: boolean
  md_text: string
  md_sha256: string
  page_id: string | null
  diff_url: string | null
  note: string | null
}

// --- 评测层 ---

export interface ResponseRow {
  sample_id: string
  dataset: string
  question: string
  answer_mode: string | null
  http_status: number
  error: string | null
  latency_ms: number | null
  requested_at: string
  answer: string | null
  retrieved_count: number
  citation_count: number
}

export interface ResponseList {
  query_layer_id: number
  label: string
  total: number
  count_by_answer_mode: Record<string, number>
  offset: number
  limit: number
  responses: ResponseRow[]
}

export interface DatasetEval {
  dataset: string
  samples_in_subset: number
  responses_evaluated: number
  http_failures: number
  missing_responses: string[]
  unmapped_page_ids: string[]
  /** 算不了的指标。**连同原因一起显示，不要画成 0。** */
  omitted_metrics: string[]
  omission_reason: string | null
  answer_mode_distribution: Record<string, number>
  stratified: { key: string; strata: Record<string, Stratum> } | null
  /** scope -> 指标。scope 是 overall / knowledge_only / stratum:<名> / judge。 */
  scopes: Record<string, Metrics>
}

export interface Stratum {
  count: number
  metrics: Metrics
  knowledge_answer_share: number
}

export interface JudgeSummary {
  metric: string
  total: number
  scored: number
  /** 失败被排除的条数。均值的分母是 scored，不是 total。 */
  excluded: number
  mean: number | null
  failures_by_kind: Record<string, number>
  failure_rate: number
}

export interface EvalLayerDetail {
  id: number
  label: string
  config_hash: string
  ks: number[]
  /** 这一轮勾了哪些指标。报告页的列以它为准。 */
  metrics: string[]
  query_layer: { id: number; label: string; score_threshold: number | null }
  datasets: DatasetEval[]
  judge: JudgeSummary
  badcase_causes: Record<string, number>
}

export interface SampleRow {
  sample_id: string
  dataset: string
  answer_mode: string | null
  ok: number
  http_status: number
  gold_count: number
  retrieved_count: number
  citation_count: number
  latency_ms: number | null
  answer: string | null
  metrics: Metrics
}

export interface SampleList {
  /** **按 answerMode 分组是默认行为**，不是筛选器。见后端 note。 */
  by_answer_mode: Record<string, SampleRow[]>
  counts: Record<string, number>
  note: string
}

export interface WorstList {
  metric: string
  higher_is_better: boolean
  samples: Array<{
    sample_id: string
    dataset: string
    value: number
    answer_mode: string | null
    answer: string | null
    root_cause: string | null
  }>
  /** 先看这个：非 knowledge 的那些是生成端回落，低分不是检索问题。 */
  count_by_answer_mode: Record<string, number>
  note: string
}

export interface JudgeVerdict {
  sample_id: string
  metric: string
  score: number | null
  /** rate_limit / timeout / parse_error / refusal。非空时 score 必然是 null。 */
  failure_kind: string | null
  reasoning: {
    claims?: Array<{ claim: string; verdict: string; evidence: string }>
    claim_count?: number
    supported?: number
    unsupported?: number
    contradicted?: number
    skipped?: string
  } | null
  prompt_version: string
}

export interface Annotation {
  id: number
  level: string
  target_id: string
  author_kind: 'human' | 'model'
  author: string
  labels: string[]
  note: string | null
  source: string
  confidence: number | null
  created_at: string
}

// --- 归因层 ---

export type RootCause =
  | 'generation_fallback'
  | 'compiled_away'
  | 'citation_dropped'
  | 'retrieval_miss'
  | 'graph_edge_missing'
  | 'gold_annotation_suspect'
  | 'unknown'

export interface BadcaseAnalysis {
  sample_id: string
  root_cause: RootCause
  labels: string[]
  evidence: {
    answer_mode: string | null
    hit: number | null
    full_coverage: number | null
    gold_count: number | null
    retrieved_count: number | null
    citation_count: number | null
    truncated_gold: number
    question_terms_lost: string[]
    graph_exclusive_gold_count: number | null
    f1: number | null
    /** false = 没配只读库，compiled_away 那条判据降级了。 */
    lineage_available: boolean
    model?: {
      contributing_factors?: string[]
      disagreement?: string | null
      confidence?: number | null
    }
    model_error?: string
  }
  narrative: string | null
  /** true = 纯规则。规则不需要模型配置就能出结果。 */
  rule_based: boolean
  remedy?: string
  analyzed_at: string
}

export interface BadcaseList {
  eval_layer_id: number
  count_by_root_cause: Record<string, number>
  analyses: BadcaseAnalysis[]
  causes: Record<string, { remedy: string }>
}

export interface BatchAnalysis {
  eval_layer_id: number
  metric: string
  analyzed: number
  count_by_root_cause: Record<string, number>
  results: Array<BadcaseAnalysis & { model_error: string | null }>
  note: string
}

export interface SampleDetail extends SampleRow {
  eval_layer_id: number
  query_layer_id: number
  index_layer_id: number
  question: string | null
  reference_answers: string[]
  metadata: Record<string, unknown>
  gold_doc_ids: string[]
  /** doc_id -> page_id。血缘视图的入口。null 表示这篇不在 page_map 里。 */
  gold_pages: Record<string, string | null>
  detail: Record<string, unknown>
  response: Record<string, unknown> | null
  judge_verdicts: JudgeVerdict[]
  annotations: Annotation[]
  badcase_analysis: BadcaseAnalysis | null
  audit: Record<string, unknown> | null
}

// --- 血缘与 diff ---

export interface Artifact {
  id: string
  title: string
  /** entity / source_summary。 */
  page_type: string
  compile_scope: string | null
  /** 实体合并的键。 */
  canonical_key: string | null
  stale_at: string | null
}

export interface Chunk {
  id: string
  knowledge_page_id: string
  title: string
  chunk_role: string | null
  retrieval_channel: string | null
  /** 换 embedding 后旧 chunk 的 profile 对不上，那些 chunk 永远召回不到。 */
  embedding_profile: string | null
  text: string
}

export interface GraphEdge {
  id: string
  /** 自由生成的取值（555 条边散在 377 种上），所以图遍历无法按类型做。 */
  relation: string
  from_title: string
  to_title: string
  stale_at: string | null
}

export interface Lineage {
  source_page_id: string
  artifacts: Artifact[]
  /** 参与召回的文本。 */
  chunks: Chunk[]
  /** 原文，**不参与召回**。 */
  source_chunks: Array<{ id: string; text: string }>
  edges: GraphEdge[]
  counts: { artifacts: number; chunks: number; source_chunks: number; edges: number }
}

export interface Diff {
  source_page_id: string
  source: { text: string; chunk_count: number; note: string }
  compiled: { text: string; chunk_count: number; artifact_count: number; note: string }
  diff: {
    source_chars: number
    compiled_chars: number
    /** > 1 是扩写。实测中位 2.19 倍 —— 丢词是改写策略，不是空间不足。 */
    expansion_ratio: number | null
    dropped: string[]
    dropped_total: number
    added: string[]
    added_total: number
    retention: number | null
  }
  /** 非空 = 这条样本的词法召回已经断了。 */
  question_terms_lost: string[]
  verdict: string
  /** 这次查的是哪个部署的库。空结果与「配置指向了别处」长得一样，靠它区分。 */
  queried?: { base_url: string }
}

export interface SampleLineage {
  sample_id: string
  question: string
  answer_mode: string | null
  metrics: Metrics
  gold: Array<{
    doc_id: string
    page_id: string | null
    error?: string
    lineage?: Lineage
    diff?: Diff
  }>
}

// --- 任务层 ---

export interface Stage {
  stage: string
  label: string
  args: string[]
  cost: string
  needs_akasha: boolean
}

export interface Task {
  id: number
  stage: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  argv: string[]
  /** 这一轮的参数。argv 里只有 run_config 的 id，所以单独给。 */
  args: Record<string, unknown>
  pid: number | null
  exit_code: number | null
  progress_done: number
  progress_total: number | null
  progress_note: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
}

export interface TaskDetail extends Task {
  events: Array<{ id: number; at: string; level: string; message: string }>
}
