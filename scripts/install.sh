#!/usr/bin/env bash
# DeepSoul Installer — 一条命令免配置安装
#
# 使用:
#   curl -fsSL https://raw.githubusercontent.com/xpailab/deepseek-SOUL/main/scripts/install.sh | bash
#   # 或本地安装
#   bash scripts/install.sh
#
# 支持: Linux, macOS, WSL2
# 依赖: Python 3.11+, ripgrep (可选), git (可选)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════╗"
echo "  ║       DeepSoul Installer        ║"
echo "  ║  下一代 AI Agent 框架一键安装   ║"
echo "  ╚══════════════════════════════════╝"
echo -e "${NC}"

# ── 检测系统 ──────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux*)   PLATFORM="linux" ;;
    Darwin*)  PLATFORM="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)        echo -e "${RED}不支持的系统: $OS${NC}"; exit 1 ;;
esac

echo -e "${CYAN}→ 检测到系统:${NC} $PLATFORM"

# ── Python 检查 ──────────────────────────────────
PYTHON=""
for cmd in python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${YELLOW}Python 3.11+ 未找到，尝试安装...${NC}"
    if [ "$PLATFORM" = "linux" ]; then
        sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv || {
            echo -e "${RED}无法安装 Python 3.11，请手动安装${NC}"
            exit 1
        }
        PYTHON="python3.11"
    elif [ "$PLATFORM" = "macos" ]; then
        if command -v brew &>/dev/null; then
            brew install python@3.11
        else
            echo -e "${RED}请安装 Homebrew 后重试，或手动安装 Python 3.11+${NC}"
            exit 1
        fi
        PYTHON="python3.11"
    else
        echo -e "${RED}请手动安装 Python 3.11+: https://python.org${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Python:${NC} $($PYTHON --version)"

# ── 安装目录 ─────────────────────────────────────
SOUL_HOME="${SOUL_HOME:-$HOME/.soul}"
SOUL_DIR="${SOUL_DIR:-$HOME/deepseek-SOUL}"

echo -e "${CYAN}→ 安装目录:${NC} $SOUL_DIR"

# 如果当前目录就是项目目录，就地安装
if [ -f "./pyproject.toml" ] && [ -d "./soul" ]; then
    SOUL_DIR="$(pwd)"
    echo -e "${CYAN}→ 检测到项目目录，就地安装${NC}"
fi

mkdir -p "$SOUL_HOME/workspace"
mkdir -p "$SOUL_HOME/skills"
mkdir -p "$SOUL_HOME/logs"

# ── 安装依赖 ─────────────────────────────────────
echo -e "${CYAN}→ 安装 Python 依赖...${NC}"

cd "$SOUL_DIR"

# 创建虚拟环境（推荐）
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo -e "${GREEN}✓ 创建虚拟环境${NC}"
fi

# 激活虚拟环境
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true

# 安装
pip install --upgrade pip -q
pip install -e ".[all]" -q 2>&1 | tail -3

echo -e "${GREEN}✓ 依赖安装完成${NC}"

# ── 生成默认配置 ─────────────────────────────────
CONFIG_FILE="$SOUL_HOME/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${CYAN}→ 生成默认配置...${NC}"

    echo -n "API Provider [deepseek]: "
    read -r API_PROVIDER
    API_PROVIDER=${API_PROVIDER:-deepseek}

    echo -n "API Key: "
    read -r API_KEY

    echo -n "Model [deepseek-v4-pro]: "
    read -r API_MODEL
    API_MODEL=${API_MODEL:-deepseek-v4-pro}

    cat > "$CONFIG_FILE" << YAML
version: "0.1.0"
llm:
  provider: $API_PROVIDER
  model: $API_MODEL
  api_key: "$API_KEY"
  api_base: ""
  max_tokens: 8192
  temperature: 0.7
lane:
  max_concurrent: 4
  session_concurrent: 1
  subagent_concurrent: 8
  cron_concurrent: 2
memory:
  workspace_dir: "$SOUL_HOME/workspace"
  fts_db_path: "$SOUL_HOME/memory.db"
  honcho_enabled: true
  predictive_enabled: true
skill:
  skills_dir: "$SOUL_HOME/skills"
  auto_generate: true
  gepa_enabled: true
gateway:
  port: 18789
  host: "0.0.0.0"
  dm_policy: "pairing"
  websocket_enabled: true
sandbox:
  default_mode: "local"
  docker_image: "soul-sandbox:latest"
  readonly_root: true
mlops:
  output_dir: "$SOUL_HOME/training"
  max_trajectories: 1000
  parallel_workers: 4
YAML

    echo -e "${GREEN}✓ 配置已保存到:${NC} $CONFIG_FILE"
else
    echo -e "${GREEN}✓ 配置文件已存在${NC}"
fi

# ── 创建默认 Prompt 文件 ─────────────────────────
if [ ! -f "$SOUL_HOME/SOUL.md" ]; then
    cat > "$SOUL_HOME/SOUL.md" << 'EOF'
# DeepSoul Agent 人格

你是一个高效、专业、友好的 AI 助手。你的核心特质：

## 性格
- 积极主动：不等待指令，主动发现并解决问题
- 严谨可靠：反复检查代码，确保没有错误
- 简洁直接：用最少的话说清楚事情
- 持续学习：从每次交互中学习改进

## 边界
- 不执行危险系统命令
- 不泄露敏感信息
- 不进行未经授权的网络操作
- 所有文件操作在允许的工作空间内

## 风格
- 默认使用中文沟通
- 技术内容保持专业准确
- 代码优先于长篇解释
EOF
    echo -e "${GREEN}✓ 创建 SOUL.md${NC}"
fi

if [ ! -f "$SOUL_HOME/workspace/MEMORY.md" ]; then
    touch "$SOUL_HOME/workspace/MEMORY.md"
    echo -e "${GREEN}✓ 创建 MEMORY.md${NC}"
fi

if [ ! -f "$SOUL_HOME/workspace/USER.md" ]; then
    touch "$SOUL_HOME/workspace/USER.md"
    echo -e "${GREEN}✓ 创建 USER.md${NC}"
fi

# ── 安装 CLI 命令 ───────────────────────────────
echo -e "${CYAN}→ 安装 CLI 命令...${NC}"

# 创建可执行脚本
SOUL_BIN="$SOUL_DIR/scripts/soul"
cat > "$SOUL_BIN" << SCRIPT
#!/usr/bin/env bash
# DeepSoul CLI Launcher
SOUL_DIR="$SOUL_DIR"
if [ -f "\$SOUL_DIR/.venv/bin/activate" ]; then
    source "\$SOUL_DIR/.venv/bin/activate"
fi
exec python -m soul.cli "\$@"
SCRIPT
chmod +x "$SOUL_BIN"

# 添加到 PATH（可选）
if [[ ":$PATH:" != *":$SOUL_DIR/scripts:"* ]]; then
    echo -e "${YELLOW}→ 将以下路径添加到 PATH 以使用 soul 命令:${NC}"
    echo -e "  ${BOLD}export PATH=\"$SOUL_DIR/scripts:\$PATH\"${NC}"
    echo ""
    # 自动添加到 shell 配置
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ]; then
            if ! grep -q "$SOUL_DIR/scripts" "$rc" 2>/dev/null; then
                echo "export PATH=\"$SOUL_DIR/scripts:\$PATH\"  # DeepSoul" >> "$rc"
            fi
        fi
    done
fi

# ── 验证安装 ─────────────────────────────────────
echo ""
echo -e "${CYAN}→ 验证安装...${NC}"

$PYTHON -c "
import soul
from soul.__init__ import __version__
print(f'DeepSoul v{__version__} 安装成功')
" && echo -e "${GREEN}✓ 核心模块加载正常${NC}"

# ── 完成 ─────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║     DeepSoul 安装完成！          ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════╝${NC}"
echo ""
echo -e "快速开始:"
echo -e "  ${CYAN}soul chat${NC}              # 交互式对话"
echo -e "  ${CYAN}soul run \"任务描述\"${NC}    # 单次执行"
echo -e "  ${CYAN}soul gateway${NC}          # 启动网关"
echo -e "  ${CYAN}soul status${NC}           # 查看状态"
echo -e "  ${CYAN}soul doctor${NC}           # 诊断检查"
echo -e "  ${CYAN}soul config --all${NC}     # 查看配置"
echo -e "  ${CYAN}soul --help${NC}           # 查看帮助"
echo ""
echo -e "Web UI:  ${CYAN}soul-gateway --port 8080${NC}"
echo -e "配置:    ${CYAN}$CONFIG_FILE${NC}"
echo -e "工作空间: ${CYAN}$SOUL_HOME/workspace${NC}"
echo ""
