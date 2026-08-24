#!/usr/bin/env python3
"""run_mini_plugin.py — mini + 插件验证 runner: 先装插件(env 门控)再跑 run_mini.main()。"""
import sys

sys.path.insert(0, "/work/plugin_a1")
import routea_plugin_a1  # noqa: F401  (env 门控安装)

sys.path.insert(0, "/work")
import run_mini

sys.argv = ["run_mini.py"] + sys.argv[1:]
run_mini.main()
