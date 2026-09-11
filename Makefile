# Akasha-Benchmark。
#
#   make            # 看有哪些目标
#   make setup      # sync -> download -> migrate -> normalize -> validate
#   make serve      # 起平台，之后所有实验从界面里跑
SHELL := /bin/sh

DATASETS ?=
ARGS     ?=

PY := uv run python
DB := akasha_bench.db

DATASET_FLAGS := $(foreach d,$(DATASETS),--dataset $(d))

.PHONY: help setup sync download check-download migrate migrate-status normalize \
        validate web serve dev test smoke distclean

help:
	@echo 'Akasha-Benchmark'
	@echo ''
	@echo '装环境与底座：'
	@echo '  sync            uv sync，装依赖（Python 3.12）'
	@echo '  download        下载四组数据到 dataset/  (~137MB)'
	@echo '  migrate         建库 / 升级 schema（迁移前自动备份）'
	@echo '  migrate-status  看迁移状态'
	@echo '  normalize       原始数据 -> 库里的 sample + corpus_doc'
	@echo '  validate        验收：逐行过全量数据，四组须全过'
	@echo '  setup           以上全做一遍'
	@echo ''
	@echo '平台：'
	@echo '  web             构建前端（npm install + build）'
	@echo '  serve           起平台，单进程单端口，只绑 127.0.0.1'
	@echo '  dev             开发模式提示（Vite dev server + FastAPI 两进程）'
	@echo ''
	@echo '其他：'
	@echo '  test            跑 pytest'
	@echo '  smoke           在线冒烟，核对接口字段约定  [需要 Akasha 在线]'
	@echo '  distclean       删导出/缓存/前端产物（不删库与 dataset/）'
	@echo ''
	@echo 'Akasha 连接（含密钥）在配置层里填，存库。项目目录不留配置文件。'

# --- 依赖与数据 ---------------------------------------------------------------

sync:
	uv sync

download:
	$(PY) scripts/download_datasets.py

check-download:
	$(PY) scripts/download_datasets.py --check

# --- 库 -----------------------------------------------------------------------

# 迁移前自动备份成 akasha_bench.db.pre-{version}。已应用的迁移被改过会报错。
migrate:
	$(PY) -m akasha_benchmark.store.migrate

migrate-status:
	$(PY) -m akasha_benchmark.store.migrate --status

# --- 归一化 -------------------------------------------------------------------
#
# 归一化留在 make 里而不是只在界面上，因为它是「库能用起来」的一部分：
# 没有它数据集那一栏是空的，界面上也就没有可选的东西。它同时也在界面上有入口。

# 依赖 check-download 而不是 download：数据已在位时不该为了跑一次归一化去连 HF。
normalize: check-download migrate
	$(PY) -m akasha_benchmark.normalize $(DATASET_FLAGS) $(ARGS)

validate: normalize
	$(PY) scripts/validate_datasets.py --db $(DB) $(DATASET_FLAGS) $(ARGS)

setup: validate
	@echo ''
	@echo 'OK 底座就绪。产物在 $(DB)。'
	@echo '   接下来：make web && make serve，然后在界面上配置 Akasha 连接并跑实验。'

# --- 平台 ---------------------------------------------------------------------

web:
	npm --prefix web install
	npm --prefix web run build

# 生产形态：单进程单端口，FastAPI 挂 web/dist。
# 不在这里做前置检查：库不存在与前端未构建两件事，akasha-platform 的 main()
# 自己就会报（分别是非零退出与首页的提示）。放两份的话，从 PowerShell 直接跑
# uv run akasha-platform 的人看不到 make 这一份，而两份迟早会漂。
serve:
	uv run akasha-platform

dev:
	@echo '开发模式要两个进程：'
	@echo '  终端 1：uv run akasha-platform            # API on :8848'
	@echo '  终端 2：npm --prefix web run dev          # Vite on :5173（/api 代理过去）'

# --- 杂项 ---------------------------------------------------------------------

test:
	uv run pytest -q

SMOKE_DATASET := $(firstword $(DATASETS) hotpotqa)
SMOKE_LABEL   ?= run002

smoke: export AKASHA_LIVE = 1
smoke: export AKASHA_LIVE_LABEL = $(SMOKE_LABEL)
smoke: export AKASHA_LIVE_DATASET = $(SMOKE_DATASET)
smoke:
	uv run pytest tests/test_live_akasha.py -v $(ARGS)

distclean:
	$(PY) scripts/clean.py all
