SHELL := /bin/sh

.PHONY: help sync serve web build test

help: ## 显示可用命令
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## 安装依赖（Python 与前端）
	uv sync
	npm --prefix web install

serve: ## 启动后端 :8848（建表在启动时自动完成）
	uv run uvicorn akasha_platform.main:app --reload --port 8848

web: ## 启动前端热更新 :5173
	npm --prefix web run dev

build: ## 构建前端产物
	npm --prefix web run build

test: ## 运行离线测试
	uv run pytest -q
