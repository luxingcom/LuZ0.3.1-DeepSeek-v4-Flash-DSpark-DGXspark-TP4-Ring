# routeb_merged_plugin setup.py — vLLM 插件包（vllm.general_plugins 注册, EngineCore 子进程自动加载）
from setuptools import find_packages, setup

setup(
    name="routeb-merged-plugin",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "routeb_merged_plugin = routeb_merged_plugin:install",
        ],
    },
    python_requires=">=3.10",
)
