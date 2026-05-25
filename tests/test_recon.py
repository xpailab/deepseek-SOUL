"""侦察阶段 + 模糊反问 + 编译验证测试。"""

from __future__ import annotations

from soul.engine.agent import Agent


class TestAmbiguityDetection:
    def test_vague_short(self):
        assert Agent._is_vague_task("修一下") is True
        assert Agent._is_vague_task("有问题") is True
        assert Agent._is_vague_task("帮我看看") is True

    def test_vague_optimize(self):
        assert Agent._is_vague_task("优化一下性能") is True
        assert Agent._is_vague_task("改一下这个") is True
        assert Agent._is_vague_task("报错了怎么办") is True

    def test_specific_with_file(self):
        assert Agent._is_vague_task("修复 soul/engine/agent.py 的空指针错误") is False
        assert Agent._is_vague_task("main.py line 42 报错 TypeError") is False

    def test_specific_with_path(self):
        assert Agent._is_vague_task("D:/projects/app/main.go 编译错误") is False

    def test_specific_with_error(self):
        assert Agent._is_vague_task("docker: Error response from daemon: port already in use") is False

    def test_specific_long(self):
        assert Agent._is_vague_task(
            "把 soul/memory/manager.py 里的 query 方法改成异步的，加上超时控制"
        ) is False

    def test_not_vague_casual(self):
        assert Agent._is_vague_task("你好") is False  # 短但不是模糊任务词
        assert Agent._is_vague_task("列出当前目录") is False  # 不包含模糊词
        assert Agent._is_vague_task("列出 /tmp 下的文件") is False  # 有具体路径
        assert Agent._is_vague_task("帮我看看性能问题") is True  # 含模糊词 + 无具体信息


class TestBuildVerification:
    def test_python_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/main.py")
        assert "py_compile" in prompt or "python" in prompt
        assert "/app/main.py" in prompt

    def test_js_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/index.js")
        assert "node --check" in prompt

    def test_go_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/main.go")
        assert "go build" in prompt or "go vet" in prompt

    def test_rust_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/main.rs")
        assert "cargo" in prompt.lower()

    def test_java_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/Main.java")
        assert "javac" in prompt or "mvn" in prompt

    def test_shell_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/deploy.sh")
        assert "bash -n" in prompt

    def test_non_code_file(self):
        prompt = Agent._build_verify_prompt("write_file", "/app/README.md")
        assert prompt == ""

    def test_no_filepath(self):
        prompt = Agent._build_verify_prompt("write_file", "")
        assert prompt == ""


class TestReconPrompt:
    def test_recon_for_specific_task(self):
        # 需要一个 Agent 实例（不需要完整初始化）
        agent = Agent.__new__(Agent)
        prompt = agent._recon_prompt("修复 soul/engine/agent.py 的空指针错误")
        assert "侦察" in prompt
        assert "摸底" in prompt or "只读" in prompt
        # 具体任务不触发反问
        assert "模糊" not in prompt

    def test_recon_for_vague_task(self):
        agent = Agent.__new__(Agent)
        prompt = agent._recon_prompt("优化一下")
        assert "模糊" in prompt
        assert "反问" in prompt
