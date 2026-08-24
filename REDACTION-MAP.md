# 脱敏映射表（REDACTION MAP）

本文件说明开源发布副本中应用的**占位符语义**（不包含任何真实敏感值，真实映射与批量替换工具保留在本地未提交）。

## 占位符语义

| 占位符 | 语义 | 示例场景 |
|---|---|---|
| `<PASSWORD>` | 明文密码/提权口令 | 运维脚本中的 sudo 密码 |
| `<KEY_PREFIX_OLD>` / `<KEY_PREFIX_NEW>` | API key 前缀（轮换前后） | key 轮换窗口报告 |
| `<API_KEY>` | API key 值 | `--api-key`、`VLLM_API_KEY` |
| `<BEARER>` | Bearer token | 网关/监控鉴权 |
| `<USER>` | 内部用户名 | `liuxiaoya` 等 |
| `<INSTALL_DIR>` | 内部安装目录 | `/opt/aicad-prod` |
| `<MODELS_DIR>` / `<HOME_DIR>` | 模型/用户主目录 | `/home/<USER>/models`、`/data/models` |
| `<NODE_IP>` | 内网 IP（端口保留） | `192.168.x.x`、`10.x.x.x` |
| `node0X` | 主机名 | `dgxspark01`~`dgxspark04` |

## 规则

1. 批量替换工具 `redact.py` + 私有模式文件 `redact-patterns.json` 仅存在于本地（`redact-patterns.json` 已 gitignore，含真实值，勿提交）。
2. 已对发布副本执行全量替换并重扫验证：**工作树与已提交树敏感模式残留 = 0**（扫描类别：内网 IP / 主机名 / 内部用户名 / 密码 / key 前缀 / 内部路径 / api key 形态）。
3. 若后续发现遗漏（如新 IP、新路径、新 key），重新运行 `redact.py` 后提交。

## 已知简化

- 主机名 `dgxspark01~04` 统一映射为 `node0X`（未保留节点序号语义），以最大化信息隐藏；如需保留 `node01`~`node04` 可在私有模式文件中调整。
- 端口号在 IP 脱敏时保留（如 `<NODE_IP>:8001`），端口不视为敏感。

*本映射表不包含真实敏感值，可随开源仓库发布。*
