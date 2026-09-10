# Akasha-Benchmark 流程。命令与 README / docs/PLAN.md §1 一致。
#
#   make              # 看有哪些目标（默认不做事：第一步要下 137MB）
#   make offline      # 本地阶段：download -> normalize -> validate -> subset
#   make smoke        # 单样本在线冒烟，验接口约定（便宜，先跑这个）
#   make ingest       # 入库，需要 Akasha 在线
#   make query        # 查询，需要 Akasha 在线
#   make report       # 评测，纯本地
#
# 常用覆盖：
#   make offline RUN_ID=run002 SEED=42
#   make normalize DATASETS=hotpotqa
#   make query RUN_ID=run001 ARGS="--limit 5"
#

SHELL := /bin/sh

RUN_ID   ?= run001
SEED     ?= 20260908
QA_LIMIT ?=
DATASETS ?=
ARGS     ?=

PY := uv run python

# DATASETS 为空时不传 --dataset，默认全量数据集。
DATASET_FLAGS := $(foreach d,$(DATASETS),--dataset $(d))
SUBSET_FLAGS  := $(if $(QA_LIMIT),--qa-limit $(QA_LIMIT),)

DATA_DIR := data

INGEST_MANIFEST   := $(DATA_DIR)/ingest/$(RUN_ID)/manifest.json
RESPONSE_MANIFEST := $(DATA_DIR)/responses/$(RUN_ID)/manifest.json
REPORT           := $(DATA_DIR)/reports/$(RUN_ID)/report.md

.PHONY: help all offline sync download check-download normalize validate \
        subset smoke ingest query report audit test clean-data distclean

help:
	@echo 'Akasha-Benchmark  (RUN_ID=$(RUN_ID))'
	@echo ''
	@echo '按顺序执行：'
	@echo '  sync            uv sync，装依赖（Python 3.12）'
	@echo '  download        下载四组数据到 dataset/  (~137MB)'
	@echo '  check-download  只校验已下载的文件，不下载'
	@echo '  normalize       归一化成 samples.jsonl + corpus.jsonl'
	@echo '  validate        验收：逐行过全量数据，四组须全过'
	@echo '  subset          抽子集，保证 gold 全覆盖'
	@echo '  smoke           单样本在线冒烟，核对接口字段约定  [需要 Akasha 在线]'
	@echo '  ingest          入库+编译+质量闸门   [需要 Akasha 在线]'
	@echo '  query           逐条跑 query 存响应   [需要 Akasha 在线]'
	@echo '  report          离线算指标出报告'
	@echo '  audit           可选    审计表归因（需 psycopg + database_url）'
	@echo ''
	@echo '组合目标：'
	@echo '  offline         download -> normalize -> validate -> subset'
	@echo '  all             offline，再提示在线阶段需要手动执行'
	@echo '  test            跑 pytest'
	@echo ''
	@echo '变量：RUN_ID SEED QA_LIMIT DATASETS ARGS'
	@echo '  例：make normalize DATASETS="hotpotqa musique"'
	@echo '      make query ARGS="--limit 5"'

# --- 依赖与数据 ---------------------------------------------------------------

sync:
	uv sync

download:
	$(PY) scripts/download_datasets.py

check-download:
	$(PY) scripts/download_datasets.py --check

# --- 归一化 -------------------------------------------------------------------

# 依赖 check-download 而不是 download：数据已在位时不该为了跑一次归一化去连 HF。
# 缺文件时 --check 以非零退出，这里就会停下，并报出缺哪几个。
normalize: check-download
	$(PY) -m akasha_benchmark.normalize $(DATASET_FLAGS) $(ARGS)

# 依赖 normalize，这样 offline 的链条是 check-download -> normalize -> validate -> subset。
# 只想单独跑验收（不重新归一化）用：make -o normalize validate
validate: normalize
	$(PY) scripts/validate_datasets.py $(DATASET_FLAGS) $(ARGS)

# --- 抽子集 -------------------------------------------------------------------

# 走 validate：子集是从归一化产物里抽的，底座没验收过就抽，抽出来的问题
# 会一路带到入库和查询，而那两步很贵。
subset: validate
	$(PY) -m akasha_benchmark.subset --run-id $(RUN_ID) --seed $(SEED) \
		$(SUBSET_FLAGS) $(DATASET_FLAGS) $(ARGS)

# --- 入库与查询：需要 Akasha 在线 ---------------------------------------------

ingest:
	@test -f $(DATA_DIR)/subsets/$(RUN_ID)/$(firstword $(DATASETS) hotpotqa)/manifest.json \
		|| { echo 'ERROR 没有 $(RUN_ID) 的子集产物，先执行：make subset RUN_ID=$(RUN_ID)' >&2; exit 1; }
	$(PY) -m akasha_benchmark.ingest --run-id $(RUN_ID) $(DATASET_FLAGS) $(ARGS)

# 入库的质量闸门在这里再拦一道。ingest 在闸门失败时已经会非零退出，
# 但那之后可以手工重跑 query；这条检查让「闸门没过就跑查询」在 make 这一层也拦住。
#
# 提示信息由 echo 输出、Python 只当一个不出声的退出码判据：Windows 上
# Python 的 stdout/stderr 编码跟随控制台（实测 gbk），从 Python 里打中文会变成乱码，
# 而 shell 的 echo 直接透传字节、显示正常。别把这段信息搬回 Python 里。
query:
	@test -f $(INGEST_MANIFEST) \
		|| { echo 'ERROR 缺 $(INGEST_MANIFEST)，先执行：make ingest RUN_ID=$(RUN_ID)' >&2; exit 1; }
	@$(PY) -c "import json,sys; sys.exit(0 if json.load(open(r'$(INGEST_MANIFEST)',encoding='utf-8')).get('quality_passed') else 1)" \
		|| { echo 'ERROR 入库质量闸门未通过，不要开始跑查询。详见 $(INGEST_MANIFEST) 的 quality / quality_passed 字段' >&2; exit 1; }
	$(PY) -m akasha_benchmark.run_queries --run-id $(RUN_ID) $(DATASET_FLAGS) $(ARGS)

# --- 评测：离线指标 -----------------------------------------------------------

report:
	@test -f $(RESPONSE_MANIFEST) \
		|| { echo 'ERROR 缺 $(RESPONSE_MANIFEST)，先执行：make query RUN_ID=$(RUN_ID)' >&2; exit 1; }
	$(PY) -m akasha_benchmark.evaluate --run-id $(RUN_ID) $(DATASET_FLAGS) $(ARGS)
	@echo '报告：$(REPORT)'

audit:
	@test -f $(DATA_DIR)/reports/$(RUN_ID)/per_sample.jsonl \
		|| { echo 'ERROR 缺评测产物，先执行：make report RUN_ID=$(RUN_ID)' >&2; exit 1; }
	$(PY) -m akasha_benchmark.audit_join --run-id $(RUN_ID) $(DATASET_FLAGS) $(ARGS)

# --- 组合与杂项 ---------------------------------------------------------------

offline: subset
	@echo ''
	@echo 'OK 本地阶段完成（RUN_ID=$(RUN_ID)）。'
	@echo '   产物：$(DATA_DIR)/normalized/  $(DATA_DIR)/subsets/$(RUN_ID)/'

all: offline
	@echo ''
	@echo '入库和查询需要 Akasha 在线，请确认配置后手动执行：'
	@echo '   make smoke  RUN_ID=$(RUN_ID)     # 先用一条样本验接口，便宜且快'
	@echo '   make ingest RUN_ID=$(RUN_ID)     # 跑完看质量闸门结果'
	@echo '   make query  RUN_ID=$(RUN_ID)     # 闸门通过后再执行'
	@echo '   make report RUN_ID=$(RUN_ID)'

test:
	uv run pytest -q

# 在线冒烟。默认 skip，所以 make test 不需要 Akasha 在线；这个目标显式开 AKASHA_LIVE=1。
# 两组用例：单样本往返核对字段约定（约 5 分钟），批量三段走真实阶段入口、
# 覆盖批量与续跑与报告产出（约 9 分钟）。只跑后者：make smoke ARGS="-k pipeline"
#
# 建的都是 smoke 前缀的独立 Space 并在跑完删掉，不碰 bench 前缀那几个。
# 产物写进 data/smoke/，单样本那趟可离线重放：
#   AKASHA_LIVE_REPLAY=data/smoke/<ts>-roundtrip.json uv run pytest tests/test_live_akasha.py
smoke:
	@test -f $(DATA_DIR)/subsets/$(RUN_ID)/$(firstword $(DATASETS) hotpotqa)/manifest.json \
		|| { echo 'ERROR 没有 $(RUN_ID) 的子集产物，先执行：make subset RUN_ID=$(RUN_ID)' >&2; exit 1; }
	AKASHA_LIVE=1 AKASHA_LIVE_RUN_ID=$(RUN_ID) \
		$(if $(DATASETS),AKASHA_LIVE_DATASET=$(firstword $(DATASETS)),) \
		uv run pytest tests/test_live_akasha.py -v $(ARGS)

# 只删本次 run 的在线产物，留着归一化底座和数据集 —— 重下 137MB 很贵。
clean-data:
	rm -rf $(DATA_DIR)/subsets/$(RUN_ID) $(DATA_DIR)/ingest/$(RUN_ID) \
		$(DATA_DIR)/responses/$(RUN_ID) $(DATA_DIR)/reports/$(RUN_ID)
	@echo '已删除 $(RUN_ID) 的子集及下游产物；normalized/ 和 dataset/ 保留。'

distclean:
	rm -rf $(DATA_DIR) .pytest_cache .ruff_cache
	@echo '已删除全部产物；dataset/ 保留（重下要 137MB，需要时手工删）。'
