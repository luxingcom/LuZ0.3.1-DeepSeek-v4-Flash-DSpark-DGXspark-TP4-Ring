#!/bin/bash
# Rex SRE 第二批精确采集：cgroup 路径修正 / smaps / 模型实目录 / 端口 / 可疑进程
echo "################## $(hostname) ##################"

echo "=== [A] vllm 相关进程（主机侧） ==="
ps -eo pid,ppid,rss,vsz,etime,comm,args --sort=-rss 2>/dev/null | grep -E "VLLM::|vllm serve|anemll" | grep -v grep | head -20

echo ""
echo "=== [B] 各 vllm 进程 cgroup 归属 ==="
for PID in $(ps -eo pid,comm 2>/dev/null | grep -E "VLLM::Worker|VLLM::EngineCore|VLLM::Engine" | awk '{print $1}'); do
  CGP=$(cat /proc/$PID/cgroup 2>/dev/null | awk -F: '{print $3}')
  echo "PID=$PID cgroup=$CGP"
done

echo ""
echo "=== [C] 找到 worker 主进程并读 smaps_rollup ==="
for PID in $(ps -eo pid,comm 2>/dev/null | grep -E "VLLM::Worker_TP" | awk '{print $1}'); do
  echo "--- Worker PID=$PID ---"
  if [ -r "/proc/$PID/smaps_rollup" ]; then
    grep -E "^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Swap|Pss_Anon|Pss_File|Referenced|Anonymous|KernelStack|PageTables|AnonHugePages|ShmemPmdMapped|FilePmdMapped|Locked)" /proc/$PID/smaps_rollup 2>/dev/null
  fi
done

echo ""
echo "=== [D] 容器真实 cgroup 内存统计（从 worker pid 定位） ==="
for PID in $(ps -eo pid,comm 2>/dev/null | grep -E "VLLM::Worker_TP" | awk '{print $1}'); do
  CGP=$(cat /proc/$PID/cgroup 2>/dev/null | awk -F: '{print $3}')
  CG="/sys/fs/cgroup${CGP}"
  echo "--- Worker PID=$PID -> $CG ---"
  [ -f "$CG/memory.current" ] && echo "memory.current = $(cat $CG/memory.current)"
  [ -f "$CG/memory.peak" ] && echo "memory.peak    = $(cat $CG/memory.peak)"
  [ -f "$CG/memory.max" ] && echo "memory.max     = $(cat $CG/memory.max)"
  [ -f "$CG/memory.swap.current" ] && echo "swap.current   = $(cat $CG/memory.swap.current)"
  [ -f "$CG/memory.swap.peak" ] && echo "swap.peak      = $(cat $CG/memory.swap.peak)"
  if [ -f "$CG/memory.stat" ]; then
    echo "--- memory.stat 关键项 ---"
    grep -E "^(anon |file |kernel_stack|slab|pagetables|shmem |file_mapped|file_dirty|sock |anon_thp|file_thp|unevictable|zswap|kernel |pgin|pgout|workingset)" "$CG/memory.stat" 2>/dev/null | head -40
  fi
done

echo ""
echo "=== [E] 模型真实目录大小 ==="
for MP in <INSTALL_DIR>/models/deepseek-v4-flash-0731 <MODELS_DIR>/deepseek-v4-flash-0731 /home/<USER>/models/deepseek-v4-flash-0731; do
  if [ -e "$MP" ]; then
    echo "--- $MP ---"
    readlink -f "$MP"
    du -shL "$MP" 2>/dev/null || du -sh "$MP" 2>/dev/null
    ls -la "$MP" 2>/dev/null | head -20
  fi
done

echo ""
echo "=== [F] vllm 容器端口发布 ==="
for CID in $(docker ps -q --filter name=vllm-tp4-rank); do
  NAME=$(docker inspect -f '{{.Name}}' $CID | sed 's|^/||')
  echo "--- $NAME Ports ---"
  docker inspect -f '{{json .NetworkSettings.Ports}}' $CID 2>/dev/null
  docker port $CID 2>/dev/null
done

echo ""
echo "=== [G] KV metrics 探测（8001/metrics + 授权头） ==="
for CID in $(docker ps -q --filter name=vllm-tp4-rank); do
  NAME=$(docker inspect -f '{{.Name}}' $CID | sed 's|^/||')
  echo "--- $NAME ---"
  # 容器内尝试 python urllib
  docker exec $CID python3 -c "
import urllib.request,os
key='<KEY_PREFIX_OLD>98d4cae30a416366729f09202b1f013a429a13679f973c09c5344594'
for p in [8001,8000,8100]:
    try:
        req=urllib.request.Request('http://127.0.0.1:%d/metrics'%p, headers={'Authorization':'Bearer '+key})
        data=urllib.request.urlopen(req,timeout=3).read().decode()
        print('PORT %d OK len=%d'%(p,len(data)))
        for line in data.splitlines():
            if any(k in line for k in ['kv_offload','cache_usage','num_preempt','eviction','kv_cache','offload','gpu_cache','cpu_cache','fs_cache','io_']):
                print(line)
        break
    except Exception as e:
        print('PORT %d ERR %s'%(p,e))
" 2>/dev/null || echo "容器内 python 探测失败"
done

echo ""
echo "=== [H] 03/04 可疑双 vllm 实例确认（各 PID 全命令） ==="
for PID in $(ps -eo pid,ppid,comm,args 2>/dev/null | grep -E "VLLM::|vllm serve|/usr/bin/vllm|vllm serve --model" | grep -v grep | awk '{print $1}'); do
  echo "### PID=$PID $(cat /proc/$PID/comm 2>/dev/null)"
  tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | head -c 400
  echo ""
  echo "   started: $(ps -o lstart= -p $PID 2>/dev/null)  ppid=$(awk '/PPid/{print $2}' /proc/$PID/status 2>/dev/null)"
  CGP=$(cat /proc/$PID/cgroup 2>/dev/null | awk -F: '{print $3}')
  echo "   cgroup: $CGP"
done

echo ""
echo "=== [I] 全容器 docker stats（含 anemll-embed） ==="
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>/dev/null

echo ""
echo "=== [J] anon 内存大头进程 top20（主机侧 RSS） ==="
ps -eo pid,rss,comm,args --sort=-rss 2>/dev/null | head -20 | awk 'NR>1{printf "PID=%s RSS=%.1fGB %s %s\n",$1,$2/1048576,$3,$4}'

echo "=== DONE $(hostname) ==="
