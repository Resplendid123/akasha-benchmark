SHELL := /bin/sh

PY := uv run python
DB := akasha_bench.db
DATASET_FLAGS = $(foreach d,$(DATASETS),--dataset $(d))

.PHONY: help setup sync download check-download migrate migrate-status normalize \
        validate web serve dev test distclean

help: ## 显示可用命令
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## 安装 Python 依赖
	uv sync

download: ## 下载原始数据集
	$(PY) scripts/download_datasets.py

check-download: ## 校验已下载的数据集
	$(PY) scripts/download_datasets.py --check

migrate: ## 初始化或升级数据库（自动备份）
	$(PY) -m akasha_benchmark.store.migrate

migrate-status: ## 查看迁移状态
	$(PY) -m akasha_benchmark.store.migrate --status

normalize: check-download migrate ## 归一化到数据库；支持 DATASETS、ARGS
	$(PY) -m akasha_benchmark.normalize $(DATASET_FLAGS) $(ARGS)

validate: normalize ## 归一化并验收数据
	$(PY) scripts/validate_datasets.py --db $(DB) $(DATASET_FLAGS) $(ARGS)

setup: validate ## 校验下载、建库、归一化、验收（需先 sync、download）
	@echo '底座就绪。运行 make web && make serve 启动平台。'

web: ## 启动前端 dev server :5173
	npm --prefix web run dev

serve: ## 启动后端 API :8848
	uv run akasha-platform

test: ## 运行离线测试
	uv run pytest -q

distclean: ## 清理导出、缓存、日志与前端产物，保留库和原始数据
	$(PY) scripts/clean.py all
