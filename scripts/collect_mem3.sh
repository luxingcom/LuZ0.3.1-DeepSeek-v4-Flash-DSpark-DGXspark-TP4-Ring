#!/bin/bash
# Rex SRE 第三批：精确 GPU/UMA 分配 + smaps（sudo 只读）
SUDO_PASS='<PASSWORD>'
echo "################## $(hostname) ##################"

echo "=== [A] nvidia-smi 精确 GPU 内存（compute apps csv） ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
echo "--- nvidia-smi 总内存 ---"
nvidia-smi -q -d MEMORY 2>/dev/null | grep -E "Total|Used|Free|Reserved" | head -12

echo ""
echo "=== [B] smaps_rollup（sudo 只读） ==="
for PID in $(ps -eo pid,comm 2>/dev/null | grep -E "VLLM::Worker_TP|VLLM::EngineCore|^ *[0-9]+ vllm$" | awk '{print $1}'); do
  COMM=$(cat /proc/$PID/comm 2>/dev/null)
  echo "--- $COMM PID=$PID ---"
  echo "$SUDO_PASS" | sudo -S sh -c "cat /proc/$PID/smaps_rollup 2>/dev/null | grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Swap|Pss_Anon|Pss_File|Referenced|Anonymous|KernelStack|PageTables|AnonHugePages|ShmemPmdMapped|FilePmdMapped|Locked|Size)'" 2>/dev/null || echo "smaps_rollup 读取失败"
done

echo ""
echo "=== [C] vllm 进程 status（VmRSS/VmSwap/VmPeak/VmHWM） ==="
for PID in $(ps -eo pid,comm 2>/dev/null | grep -E "VLLM::Worker_TP|VLLM::EngineCore|^ *[0-9]+ vllm$" | awk '{print $1}'); do
  COMM=$(cat /proc/$PID/comm 2>/dev/null)
  echo "--- $COMM PID=$PID ---"
  grep -E "^(VmPeak|VmSize|VmLck|VmPin|VmHWM|VmRSS|RssAnon|RssFile|RssShmem|VmData|VmStk|VmExe|VmLib|VmPTE|VmSwap|Threads)" /proc/$PID/status 2>/dev/null
done

echo ""
echo "=== [D] rank0 完整相关 metrics（gpu cache / weights / scheduler） ==="
CID=$(docker ps -q --filter name=vllm-tp4-rank0)
docker exec $CID python3 -c "
import urllib.request
key='<KEY_PREFIX_OLD>98d4cae30a416366729f09202b1f013a429a13679f973c09c5344594'
req=urllib.request.Request('http://127.0.0.1:8001/metrics', headers={'Authorization':'Bearer '+key})
data=urllib.request.urlopen(req,timeout=5).read().decode()
import re
keys=['gpu_cache','cpu_cache','kv_cache','kv_offload_cpu_cache_usage','kv_offload_cpu_cache_write','kv_offload_cpu_cache_read','allocation_failure','engine_sleep','model_weights','weights_size','num_requests','num_running','num_waiting','num_swapped','scheduler','prompt','generation_tokens','running','waiting','num_preempt','max_num_seqs','long_prefill']
for line in data.splitlines():
    if any(k in line for k in keys) and not line.startswith('#'):
        print(line)
" 2>/dev/null | head -80

echo ""
echo "=== [E] 全节点 free 现状复查（1秒） ==="
free -m
echo "--- MemAvailable 连续 3 次 ---"
for i in 1 2 3; do grep -E "MemFree|MemAvailable|Cached|AnonPages" /proc/meminfo; sleep 1; done

echo ""
echo "=== [F] cgroup swap 与 host swap 总量核对 ==="
CID=$(docker ps -q --filter name=vllm-tp4-rank | head -1)
CG="/sys/fs/cgroup/system.slice/docker-$CID.scope"
echo "vllm cgroup swap.current = $(cat $CG/memory.swap.current 2>/dev/null)"
echo "host SwapTotal/SwapFree = $(grep -E 'SwapTotal|SwapFree|SwapCached' /proc/meminfo | tr '\n' ' ')"
echo "swapon: $(swapon --show 2>/dev/null | tail -1)"

echo "=== DONE $(hostname) ==="
