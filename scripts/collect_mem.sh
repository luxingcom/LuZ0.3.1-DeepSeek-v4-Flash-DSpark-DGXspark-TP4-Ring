#!/bin/bash
# 只读采集 DGX Spark TP4 节点内存构成（Rex SRE）— 不修改任何配置
echo "################## $(hostname) ##################"

echo "=== [1] free -m ==="
free -m

echo "=== [2] /proc/meminfo 关键字段 ==="
grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active|Inactive|Active\(anon\)|Inactive\(anon\)|Active\(file\)|Inactive\(file\)|Unevictable|Mlocked|SwapTotal|SwapFree|Dirty|Writeback|AnonPages|Mapped|Shmem|KReclaimable|SReclaimable|SUnreclaim|KernelStack|PageTables|NFS_Unstable|Bounce|WritebackTmp|CommitLimit|Committed_AS|VmallocUsed|Percpu|HardwareCorrupted|AnonHugePages|ShmemHugePages|ShmemPmdMapped|FileHugePages|FilePmdMapped|HugePages_Total|HugePages_Free|HugePages_Rsvd|HugePages_Surp|Hugepagesize|Hugetlb)" /proc/meminfo

echo "=== [3] swap 设备 ==="
swapon --show 2>/dev/null || echo "no swap device"
cat /proc/swaps 2>/dev/null || true

echo "=== [4] docker stats（全容器） ==="
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}"

echo "=== [5] vllm 容器 ID 与 cgroup ==="
for CID in $(docker ps -q --filter name=vllm-tp4-rank); do
  NAME=$(docker inspect -f '{{.Name}}' $CID | sed 's|^/||')
  echo "--- $NAME CID=$CID ---"
  docker inspect -f 'Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}} Cpuset={{.HostConfig.CpusetCpus}} ShmSize={{.HostConfig.ShmSize}} Memlock={{.HostConfig.Ulimits}}' $CID 2>/dev/null
  if [ -d "/sys/fs/cgroup/system.slice/docker-$CID.scope" ]; then
    CG="/sys/fs/cgroup/system.slice/docker-$CID.scope"
    echo "CGROUP_V2: $CG"
    echo "memory.current   = $(cat $CG/memory.current 2>/dev/null)"
    echo "memory.peak      = $(cat $CG/memory.peak 2>/dev/null)"
    echo "memory.max       = $(cat $CG/memory.max 2>/dev/null)"
    echo "memory.swap.current = $(cat $CG/memory.swap.current 2>/dev/null)"
    echo "memory.swap.max  = $(cat $CG/memory.swap.max 2>/dev/null)"
    grep -E "^(anon|file|kernel_stack|slab|pagetables|shmem|file_mapped|file_dirty|sock|anon_thp|file_thp|unevictable|zswap)" $CG/memory.stat 2>/dev/null
    # CPU 计数
    echo "cpu.stat:"
    cat $CG/cpu.stat 2>/dev/null
  elif [ -d "/sys/fs/cgroup/memory/docker/$CID" ]; then
    CG="/sys/fs/cgroup/memory/docker/$CID"
    echo "CGROUP_V1: $CG"
    echo "usage_in_bytes    = $(cat $CG/memory.usage_in_bytes 2>/dev/null)"
    echo "max_usage_in_bytes= $(cat $CG/memory.max_usage_in_bytes 2>/dev/null)"
    echo "limit_in_bytes    = $(cat $CG/memory.limit_in_bytes 2>/dev/null)"
    echo "kmem_usage_in_bytes = $(cat $CG/memory.kmem.usage_in_bytes 2>/dev/null)"
    grep -E "^(cache|rss|rss_huge|mapped_file|shmem|pagetables|kernel_stack|slab|sock|anon_pages|file_pages|unevictable|swap)" $CG/memory.stat 2>/dev/null
  else
    echo "cgroup 路径未找到（CID=$CID）"
    ls /sys/fs/cgroup/ 2>/dev/null | head
  fi
done

echo "=== [6] 模型目录大小 ==="
du -sh <INSTALL_DIR>/models/deepseek-v4-flash-0731 2>/dev/null || echo "模型目录不可读"
ls -la <INSTALL_DIR>/models/deepseek-v4-flash-0731 2>/dev/null | head -30

echo "=== [7] vllm 容器内进程与 smaps_rollup ==="
for CID in $(docker ps -q --filter name=vllm-tp4-rank); do
  NAME=$(docker inspect -f '{{.Name}}' $CID | sed 's|^/||')
  echo "--- $NAME 容器内 top 内存进程 ---"
  docker top $CID 2>/dev/null | awk 'NR<=2 || $6!="RSS"{print}' | head -20
  echo "--- 容器内 ps（vllm/python 主进程） ---"
  docker exec $CID sh -c 'ps -eo pid,rss,comm,args --sort=-rss 2>/dev/null | head -8' 2>/dev/null || echo "exec 失败"
  # 从主机侧取 vllm 主进程 smaps_rollup
  MAINPID=$(docker top $CID 2>/dev/null | awk 'NR>1 && ($0 ~ /vllm|python|entrypoint/){print $1}' | head -1)
  echo "主进程 PID(主机侧)=$MAINPID"
  if [ -n "$MAINPID" ] && [ -r "/proc/$MAINPID/smaps_rollup" ]; then
    echo "--- smaps_rollup PID=$MAINPID ---"
    grep -E "^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Swap|Pss_Anon|Pss_File|Referenced|Anonymous|KernelStack|PageTables|AnonHugePages|ShmemPmdMapped|FilePmdMapped)" /proc/$MAINPID/smaps_rollup 2>/dev/null
  else
    echo "smaps_rollup 不可读或无主进程"
    docker top $CID 2>/dev/null | head -10
  fi
done

echo "=== [8] nvidia-smi 输出 ==="
nvidia-smi 2>&1 | head -40 || echo "nvidia-smi 不可用"

echo "=== [9] KV 卸载相关 ==="
for CID in $(docker ps -q --filter name=vllm-tp4-rank); do
  NAME=$(docker inspect -f '{{.Name}}' $CID | sed 's|^/||')
  echo "--- $NAME 监听端口 ---"
  docker exec $CID sh -c 'ss -tlnp 2>/dev/null | head -10 || netstat -tlnp 2>/dev/null | head -10' 2>/dev/null || echo "ss 不可用"
  echo "--- $NAME kv_offload metrics 探测 ---"
  docker exec $CID sh -c 'for p in 8000 8001 8002 8100; do curl -s -m 3 http://127.0.0.1:$p/metrics 2>/dev/null | grep -iE "kv_offload|cache_usage|gpu_cache|num_preemptions|evictions" | head -20 && break; done' 2>/dev/null || echo "metrics 不可达"
done

echo "=== [10] kvssd 落盘占用 ==="
du -sh /opt/aicad-kvssd 2>/dev/null || echo "kvssd 目录不可读"
ls -la /opt/aicad-kvssd 2>/dev/null | head -15
df -h /opt/aicad-kvssd 2>/dev/null

echo "=== [11] 顶层内存占用进程（主机侧） ==="
ps -eo pid,ppid,rss,comm,args --sort=-rss 2>/dev/null | head -15 | awk '{printf "%s %s %sMB %s %s\n",$1,$2,$3/1024,$4,$5}'
echo "=== DONE $(hostname) ==="
