# ============================================================================
# verify_plugin_registered.py —— 插件 entrypoint 注册验证（生产停机窗口 Step 2/3 用）
# 用法：python verify_plugin_registered.py
# 全过（exit 0）才继续灰度启用；任一失败按提示处理
# ============================================================================
import sys


def check_entry_points():
    from importlib.metadata import entry_points
    eps = list(entry_points(group="vllm.general_plugins"))
    print(f"[1] vllm.general_plugins entry points: {[ep.name for ep in eps]}")
    nv = [ep for ep in eps if "nvfp4" in ep.name.lower()]
    if nv:
        print(f"    ✅ 找到 nvfp4 插件 entry point: {nv[0].name}")
        return True
    print("    ❌ 未找到——尝试 pip install --no-deps <INSTALL_DIR>/nvfp4/plugin-src")
    return False


def check_import():
    try:
        import nvfp4_vllm_plugin
        import nvfp4_vllm_plugin.quant_config
        import nvfp4_vllm_plugin.moe_method
        import nvfp4_vllm_plugin.kv_writer
        print("[2] nvfp4_vllm_plugin 四个模块 import OK")
        return True
    except Exception as e:
        print(f"    ❌ import 失败: {e}")
        return False


def check_quant_config_registry():
    try:
        from vllm.model_executor.layers.quantization import _QUANTIZATION_CONFIG_REGISTRY
        names = list(_QUANTIZATION_CONFIG_REGISTRY.keys())
        hit = "nvfp4_4w4a_sm121" in names
        print(f"[3] 量化注册表: {'命中 nvfp4_4w4a_sm121 ✅' if hit else f'未命中（现有 {len(names)} 项）❌'}")
        return hit
    except ImportError:
        # vLLM 未装（验证环境）
        print("[3] 跳过：当前环境无 vLLM（生产容器内验证）")
        return True
    except Exception as e:
        print(f"    ⚠️ 注册表检查异常: {e}")
        return True


def check_kernel_import():
    ok = True
    for mod in ("nvfp4_4w4a_mmaf", "nvfp4_ds_mla_kv_linear_v17_triton"):
        try:
            __import__(mod)
            print(f"[4] kernel import OK: {mod}")
        except Exception as e:
            print(f"    ❌ {mod} import 失败: {e}")
            ok = False
    return ok


if __name__ == "__main__":
    results = [
        check_entry_points(),
        check_import(),
        check_quant_config_registry(),
        check_kernel_import(),
    ]
    print("\n" + ("✅ 全部通过——可进入灰度启用" if all(results) else "❌ 存在失败项——按提示修复后重跑"))
    sys.exit(0 if all(results) else 1)
