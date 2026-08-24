# routea_plugin_a1 setup.py —— vLLM 插件包（vllm.general_plugins 注册）
# TP4 部署: pip install -e <INSTALL_DIR>/nvfp4/plugin-a1/（或 PYTHONPATH + 显式 import）
from setuptools import find_packages, setup

setup(
    name="routea-plugin-a1",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "routea_plugin_a1 = routea_plugin_a1",
        ],
    },
    python_requires=">=3.10",
)
