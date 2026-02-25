@echo off
REM TimeSeriesMCP 一键部署脚本 (Windows)

echo ==========================================
echo   TimeSeriesMCP 一键部署脚本
echo ==========================================

REM 检查Python版本
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.10+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [1/6] Python版本: %PYTHON_VERSION%

REM 检查uv是否安装
where uv >nul 2>&1
if errorlevel 1 (
    echo [2/6] 安装uv包管理器...
    pip install uv
) else (
    echo [2/6] uv已安装
)

REM 创建虚拟环境
echo [3/6] 创建虚拟环境...
call uv venv

REM 安装依赖
echo [4/6] 安装项目依赖...
call uv sync

REM 运行服务器
echo [5/6] 启动MCP服务器...
echo [6/6] 服务器已启动!
echo.
echo ==========================================
echo   服务器地址: http://127.0.0.1:12346
echo   健康检查: http://127.0.0.1:12346/health
echo ==========================================
echo.
echo 按 Ctrl+C 停止服务器

REM 启动服务器
call uv run server.py

pause
