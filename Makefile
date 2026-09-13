SHELL := /bin/sh

PY := uv run python

.PHONY: help setup sync migrate web serve test clean

help: ## 显示可用命令
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## 安装 Python 依赖
	uv sync

migrate: ## 初始化数据库
	$(PY) -m akasha_benchmark.store.migrate

setup: migrate ## 初始化数据库（需先 sync）
	@echo '底座就绪。在两个终端分别运行 make web 和 make serve。'

web: ## 启动前端热更新服务 :5173
	npm --prefix web run dev

serve: ## 启动后端热更新服务 :8848
	uv run akasha-platform --reload

test: ## 运行离线测试
	uv run pytest -q

clean: ## 清理导出、缓存、日志与前端产物，保留库和原始数据
	$(PY) scripts/clean.py all
