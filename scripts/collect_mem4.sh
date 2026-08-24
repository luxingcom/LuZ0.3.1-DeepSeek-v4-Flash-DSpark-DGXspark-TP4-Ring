#!/bin/bash
# Rex SRE 第四批：vLLM 启动内存分解 / shm / NCCL / 轻载前基线
echo "################## $(hostname) ##################"

echo "=== [A] vLLM 启动日志内存相关（docker logs，只读） ==="
CID=$(docker ps -q --filter name=vllm-tp4-rank | head -1)
docker logs $CID 2>&1 | grep -iE "KV cache|GPU memory|blocks|concurr|weights|device mem|maximum concurrency|cuda graph|cudagraph|memory pool|available" | tail -40

echo ""
echo "=== [B] 容器内 /dev/shm 与共享内存 ==="
CID=$(docker ps -q --filter name=vllm-tp4-rank | head -1)
docker exec $CID sh -c 'df -h /dev/shm; echo "---"; ls /dev/shm 2>/dev/null | head; echo "--- ipc ---"; ipcs -m 2>/dev/null | head -20' 2>/dev/null || echo "exec 失败"

echo ""
echo "=== [C] NCCL/IB 设备 ==="
ls /sys/class/infiniband/ 2>/dev/null || echo "no infiniband class"
ibstat 2>/dev/null | head -10 || echo "ibstat 不可用"
cat /sys/class/infiniband/*/ports/*/phys_state 2>/dev/null || true
echo "--- NCCL env ---"
docker exec $CID sh -c 'env | grep -iE "NCCL|CUDA" | head -20' 2>/dev/null || true

echo ""
echo "=== [D] 基线 free（轻载测试前） ==="
free -m
grep -E "MemFree|MemAvailable|AnonPages|Cached|SwapCached" /proc/meminfo

echo ""
echo "=== [E] 容器内 /tmp 验证脚本存在性 ==="
docker exec $CID sh -c 'ls -la /tmp/*.py 2>/dev/null; echo "---"; head -30 /tmp/verify_65536.py 2>/dev/null; echo "===conc1==="; head -30 /tmp/verify_conc3_65536.py 2>/dev/null' 2>/dev/null || echo "脚本查询失败"

echo "=== DONE $(hostname) ==="
