#!/bin/bash
# TimeSeriesMCP 一键部署脚本

set -e

echo "=========================================="
echo "  TimeSeriesMCP 一键部署脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "[1/6] 检查Python版本: $python_version"

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "[2/6] 安装uv包管理器..."
    pip install uv
else
    echo "[2/6] uv已安装"
fi

# 创建虚拟环境
echo "[3/6] 创建虚拟环境..."
uv venv

# 安装依赖
echo "[4/6] 安装项目依赖..."
uv sync

# 运行服务器
echo "[5/6] 启动MCP服务器..."
echo "[6/6] 服务器已启动!"
echo ""
echo "=========================================="
echo "  服务器地址: http://127.0.0.1:12346"
echo "  健康检查: http://127.0.0.1:12346/health"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止服务器"

# 启动服务器
uv run server.py
