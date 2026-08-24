# ============================================================================
# nvfp4_vllm_plugin/setup.py —— vLLM 插件包（vllm.general_plugins 注册）
# 作用：kernel① v15（prefill 4W4A）+ kernel② v17（KV 写回）纳入推理工作流
# 安装：pip install -e .   （零侵入，不改 vLLM fork）
# ============================================================================
from setuptools import setup, find_packages

setup(
    name="nvfp4-vllm-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "triton", "vllm>=0.26"],
    entry_points={
        "vllm.general_plugins": [
            "nvfp4 = nvfp4_vllm_plugin",
        ],
    },
    python_requires=">=3.10",
)
