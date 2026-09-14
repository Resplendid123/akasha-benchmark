export type TaskStatus = 'queued' | 'running' | 'paused' | 'succeeded' | 'failed'
export type RunStatus = 'running' | 'paused' | 'succeeded' | 'failed'

export interface Stage {
  stage: string
  label: string
  params: string[]
  cost: string
  needs_akasha: boolean
}

export interface Task {
  id: number
  stage: string
  status: TaskStatus
  params: Record<string, unknown>
  target_kind: string | null
  target_id: number | null
  progress_done: number
  progress_total: number | null
  progress_note: string | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface AuditEntry {
  id: number
  task_id: number | null
  stage: string
  level: string
  message: string
  at: string
}

export interface TaskDetail extends Task {
  logs: AuditEntry[]
}

// --- 配置 ---

export interface Connection {
  base_url: string
  email: string
  password: string
  database_url: string
  timeout_seconds: number
  concurrency: number
  request_interval_seconds: number
  poll_interval_seconds: number
  poll_timeout_seconds: number
  updated_at: string
  compiles: { id: number; run_id: string; workspace_id: string | null; space_id: string }[]
}

export interface ConnectionTest {
  ok: boolean
  user: { id: string | null; email: string | null; role: string | null }
  workspace: { id: string | null; name: string | null }
  is_owner: boolean
  blocked_compiles: { id: number; run_id: string; reason: string }[]
  owner_warning: string | null
  model_configs: unknown
}

export interface ModelConfig {
  feature: string
  model?: string
  baseUrl?: string
  parameters?: Record<string, unknown>
  apiKeySet?: boolean
}

export interface ModelConfigsView {
  features: string[]
  live: { configs?: ModelConfig[] } | ModelConfig[]
  compiles: { id: number; run_id: string; drift: Record<string, boolean> }[]
}

export interface Provider {
  id: number
  role: 'judge' | 'attribution'
  label: string
  base_url: string
  model: string
  api_key_set: boolean
  updated_at: string
}

// --- 数据集层与归一化层 ---

export interface DatasetFile {
  dataset: string
  kind: 'qa' | 'corpus'
  file: string
  present: boolean
  size_bytes: number | null
  rows: number | null
  error: string | null
}

export interface DatasetEntry {
  name: string
  adapter: string
  provides: string[]
  identity_rules: Record<string, string>
  expected_qa_rows: number | null
  files: DatasetFile[]
  files_ready: boolean
  normalized: boolean
  normalized_at: string | null
  qa_rows: number | null
  corpus_rows: number | null
}

export interface Sample {
  sample_id: string
  dataset: string
  dataset_sample_id: string
  question: string
  answers: string[]
  gold_doc_ids: string[]
  metadata: Record<string, unknown>
}

export interface SampleDetail extends Sample {
  gold_docs: { doc_id: string; title: string; text: string }[]
  missing_gold_doc_ids: string[]
}

export interface Paged {
  total: number
  offset: number
  limit: number
}

export interface SampleList extends Paged {
  dataset: string
  samples: Sample[]
}

export interface CorpusList extends Paged {
  dataset: string
  docs: { doc_id: string; title: string; text: string; truncated: boolean }[]
}

export interface RawSamples extends Paged {
  dataset: string
  kind: 'qa' | 'corpus'
  source_file: string
  rows: Record<string, unknown>[]
}

// --- 指标 ---

export interface MetricDefinition {
  name: string
  family: string
  requires: string[]
  kind: 'deterministic' | 'judge'
  higher_is_better: boolean
  per_k: boolean
  description: string
}

export interface MetricsView {
  definitions: MetricDefinition[]
  per_dataset: Record<string, string[]>
  computable_for_all: string[]
  computable_for_some: string[]
}

// --- 编译 / 查询 / 评测 / 归因 ---

export interface AttributionRun {
  id: number
  name: string
  eval_id: number
  metric: string
  sample_limit: number
  provider_id: number | null
  status: RunStatus
  created_at: string
  finished_at: string | null
}

export interface EvalRun {
  id: number
  name: string
  query_id: number
  ks: number[]
  metrics: string[]
  judge_provider_id: number | null
  status: RunStatus
  created_at: string
  finished_at: string | null
  attributions: AttributionRun[]
}

export interface QueryStats {
  dataset: string
  responses: number
  failures: number
  latency_mean: number | null
  latency_max: number | null
}

export interface QueryRun {
  id: number
  name: string
  compile_id: number
  score_threshold: number | null
  concurrency: number
  status: RunStatus
  created_at: string
  finished_at: string | null
  stats: Record<string, QueryStats>
  evals: EvalRun[]
}

export interface CompileStats {
  dataset: string
  docs?: number
  imported?: number
  gold?: number
  samples?: number
}

export interface Readiness {
  ready: boolean
  reasons: string[]
}

export interface CompileRun {
  id: number
  run_id: string
  datasets: string[]
  seed: number
  qa_limit: number
  negatives_ratio: number
  space_id: string | null
  space_name: string | null
  workspace_id: string | null
  status: RunStatus
  created_at: string
  finished_at: string | null
  stats: Record<string, CompileStats>
  quality: { passed: boolean; gates: Record<string, number | null> } | null
  readiness: Readiness
  queries: QueryRun[]
}

export interface CompileDoc {
  dataset: string
  doc_id: string
  is_gold: number
  page_id: string | null
  error: string | null
}

export interface QueryResponseRow {
  sample_id: string
  dataset: string
  question: string
  http_status: number
  latency_ms: number | null
  error: string | null
  answer_mode: string | null
  requested_at: string
  answer: string | null
  retrieved_count: number
  citation_count: number
}

export interface ResponseList extends Paged {
  query_id: number
  count_by_answer_mode: Record<string, number>
  responses: QueryResponseRow[]
}

export type Metrics = Record<string, number>

export interface DatasetEvalSummary {
  dataset: string
  responses_evaluated: number
  http_failures: number
  omitted_metrics: string[]
  answer_modes: Record<string, number>
  scopes: Record<string, Metrics>
}

export interface JudgeSummary {
  total: number
  scored: number
  failed: number
  mean: number | null
  failure_rate: number
  failures_by_kind: Record<string, number>
}

export interface EvalDetail {
  id: number
  name: string
  query_id: number
  ks: number[]
  metrics: string[]
  status: RunStatus
  created_at: string
  finished_at: string | null
  query: { id: number; name: string; compile_id: number }
  datasets: DatasetEvalSummary[]
  judge: JudgeSummary
}

export interface EvalSampleRow {
  sample_id: string
  dataset: string
  http_status: number
  answer: string
  question: string | null
  metrics: Metrics
}

export interface EvalSamples {
  by_answer_mode: Record<string, EvalSampleRow[]>
  counts: Record<string, number>
}

export interface WorstList {
  metric: string
  higher_is_better: boolean
  samples: {
    sample_id: string
    dataset: string
    value: number
    answer_mode: string | null
    answer: string | null
  }[]
  count_by_answer_mode: Record<string, number>
}

export interface EvalSampleDetail {
  sample_id: string
  dataset: string
  answer_mode: string | null
  http_status: number
  answer: string | null
  detail: Record<string, unknown>
  metrics: Metrics
  eval_id: number
  query_id: number
  compile_id: number
  response: Record<string, unknown> | null
  gold_pages: Record<string, string | null>
  judge_verdict: {
    score: number | null
    failure_kind: string | null
    detail: Record<string, unknown> | null
  } | null
}

export type RootCause =
  | 'generation_fallback'
  | 'compiled_away'
  | 'citation_dropped'
  | 'retrieval_miss'
  | 'graph_edge_missing'
  | 'gold_annotation_suspect'
  | 'unknown'

export interface AttributionResult {
  sample_id: string
  dataset: string
  root_cause: RootCause
  evidence: Record<string, unknown>
  narrative: string | null
  rule_based: number
  remedy: string | null
}

export interface AttributionDetail extends AttributionRun {
  count_by_root_cause: Record<string, number>
  results: AttributionResult[]
  remedies: Record<string, string>
}

export interface Lineage {
  source_page_id: string | null
  source: { text: string; chunk_count: number; note: string }
  compiled: { text: string; chunk_count: number; artifact_count: number; note: string }
  diff: {
    expansion_ratio: number | null
    retention: number | null
    dropped: string[]
    dropped_total: number
    added: string[]
    added_total: number
  }
  question_terms_lost: string[]
  verdict: string
  base_url: string
}
