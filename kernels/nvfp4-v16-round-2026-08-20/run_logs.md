# v16 轮执行日志（2026-08-20 00:33–00:55 GMT+8）

## 输入
- `nvfp4-kernels-delivery(4).zip` → 容器 `/vllm-workspace/nvfp4-delivery-v16/`
- 变更：kernel① v16.1（E2M1→fp8 e4m3 路径，D1/scale 修正）；kernel② 无新版本（维持 v11）；CUTLASS 骨架保留但 round12 已判改道 PR #42209

## 关键发现链（kernel① v16）
1. **as-shipped 编译失败**：`dot_scaled() got multiple values for argument 'lhs_format'` —— 签名核对（core.py）确认 v16 位置参数错位（w_fp8 放进了 lhs_format 槽）
2. **修复 F1**（位置参数重排）→ 下一错误：`only mxfp4 inputs can be packed along a dimension different than K` → semantic.py `assert lhs_k_pack or lhs_format=="e2m1"`：**e4m3 必须 k_pack=True**
3. **修复 F2**（lhs/rhs_k_pack=True）→ 误加 F3（scale 16 组）→ `lhs_scale must be shape [64,2]. Got [64,4]` → verify_scaled_shape：**uint8 e8m0 scale 因子=32（fp8e4nv 类型才是 16）→ F3 回退，v16 原 scale 设计正确**
4. **最终修复 = F1+F2 两行**：8/8 PASS，max_abs_err=0.000000（D1、/6、e4m3 无损全验证）
5. **性能 0.1~0.2 TFLOPS**（torch fp32 对照 17.1~18.4 同环境）→ SASS 100,289 行 / PTX 全 .bf16 mma + cvt.rn.f16x2.e4m3x2 → **e4m3→fp16 转换 + bf16 MMA + 标量 scale 模拟，无原生 FP8 MMA**

## kernel② 性能终值（4680B/token）
- v11（维持）：53.6~60.6 GB/s；v12.1：17.0~19.0；v15：9.3~10.5

## 结论
- v16.1 不可部署（性能崩塌）；kernel① 维持 v15（bf16 MMA 26.7~81.4）；400 目标 = Triton 3.7+ 或 PR #42209
- SASS 门禁升级：`mma.*e4m3|tcgen05` 才算原生 FP8
- 生产 4 rank healthy GPU 0% 未恢复

## 已落盘
- 容器：`/vllm-workspace/nvfp4-delivery-v16/`（evidence/ + v16_fixed 模块 + 全部脚本）
- 本地：`deliverables/engineering-assurance/nvfp4-v16-round-2026-08-20/`
