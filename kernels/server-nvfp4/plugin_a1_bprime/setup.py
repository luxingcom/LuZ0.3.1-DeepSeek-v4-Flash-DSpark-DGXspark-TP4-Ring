# routea_plugin_a1_bprime setup.py —— b′ native 共享插件包（vllm.general_plugins 注册）
# 部署: PYTHONPATH 指向本包父目录 + 显式 import，或 pip install -e。
# 注意: 与 plugin_a1 读同一 env VLLM_MOE_W4A4 且都 patch 类解析——两插件
# 不得同时经 entry point 激活（见包 __init__ 文档"部署注意"）。
from setuptools import find_packages, setup

setup(
    name="routea-plugin-a1-bprime",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "routea_plugin_a1_bprime = routea_plugin_a1_bprime:install",
        ],
    },
    python_requires=">=3.10",
)
