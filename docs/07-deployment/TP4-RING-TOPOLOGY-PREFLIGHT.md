# Switchless TP=4 Ring：线序识别与 NCCL HCA Map 验证说明

这份文档总结无交换机、四节点直连 RoCE Ring 上部署 TP=4 时，如何识别物理线序、确定软件 rank，并生成与现场一致的 NCCL 静态 HCA map。

节点统一使用匿名标签 `A/B/C/D`，不依赖主机名、管理网 IP 或设备序列号。

## 1. 先建立设备与地址清单

每台节点分别记录：

- 管理网接口：用于 SSH、GLOO、TCPStore；
- HCA 顺序：例如 `HCA0,HCA1,HCA2,HCA3`；
- 每个 HCA 的 netdev 名称；
- RoCE GID 类型和 index；
- 节点本地 GPU/容器可见性。

建议使用以下只读命令收集信息：

```bash
hostname
ip -br addr
ibdev2netdev
rdma link show
show_gids                         # 若安装了 perftest/rdma-core 工具
ls -l /sys/class/infiniband/*/ports/*/gids
```

不要仅凭 HCA 名称或 `NCCL_IB_HCA` 字符串顺序推断物理接线。必须同时记录“设备 index → 实际 netdev → GID”。

## 2. 用 RDMA 点到点测试识别物理边

`ping` 或 jumbo ping 只能证明 IP/MTU 路径正常，不能证明 NCCL 选择了正确的 RDMA rail。每条候选边应在两个端点分别指定 HCA，逐对测试，避免多个测试互相干扰。

服务端（节点 X）：

```bash
ib_write_bw \
  -d <SERVER_HCA> \
  -x <GID_INDEX> \
  -R \
  -s 65536 \
  -D 10 \
  --report_gbits
```

客户端（节点 Y）：

```bash
ib_write_bw <SERVER_ROCE_ADDRESS> \
  -d <CLIENT_HCA> \
  -x <GID_INDEX> \
  -R \
  -s 65536 \
  -D 10 \
  --report_gbits
```

其中：

- `-R` 使用 RDMA-CM；
- `-x` 指定 GID index；
- `-s 65536` 固定消息大小，便于比较；
- `-D 10` 运行 10 秒；
- 双向方向都要测试；
- 必须记录端点 HCA、GID、链路类型和平均带宽。

结果整理成表：

| 物理边 | 节点 X HCA | 节点 Y HCA | 双向结果 | 结论 |
|---|---|---|---:|---|
| A ↔ B | h? | h? | ? Gb/s | PASS/FAIL |
| B ↔ C | h? | h? | ? Gb/s | PASS/FAIL |
| C ↔ D | h? | h? | ? Gb/s | PASS/FAIL |
| D ↔ A | h? | h? | ? Gb/s | PASS/FAIL |

只有点到点测试确认的边，才能进入 NCCL Ring map。物理边测试通过并不等于 NCCL collective 已通过。

## 3. 从物理图选择唯一的软件 rank

物理 Ring 确认后，选择一套并永久记录的软件编号。例如：

```text
rank 0 = A
rank 1 = B
rank 2 = C
rank 3 = D
ring   = 0 → 1 → 2 → 3 → 0
```

节点标签可以旋转或反向，但同一套部署中必须保持一致。不能一部分脚本使用 `A/B/C/D`，另一部分脚本使用另一种排列。

推荐在配置中同时写出：

```text
RANK0_HOST=A
RANK1_HOST=B
RANK2_HOST=C
RANK3_HOST=D
MASTER_ADDR=<RANK0_MANAGEMENT_IP>
MASTER_PORT=<DEDICATED_CONTROL_PORT>
```

rank 是软件通信标签，不是 HCA index，也不是主机名中的数字。

## 4. 把物理边转换成静态 HCA map

假设第 2 步得到如下结果：

```text
A ↔ B  使用 HCA 0/2
B ↔ C  使用 HCA 1/3
C ↔ D  使用 HCA 0/2
D ↔ A  使用 HCA 1/3
```

在 `rank 0=A, rank 1=B, rank 2=C, rank 3=D` 的前提下，map 应表达为：

```text
0 → 1 : {0,2}    0 → 3 : {1,3}
1 → 0 : {0,2}    1 → 2 : {1,3}
2 → 1 : {1,3}    2 → 3 : {0,2}
3 → 2 : {0,2}    3 → 0 : {1,3}
```

实际项目中应由测得的边表生成该矩阵，而不是手工猜测。

关键注意事项：

1. 静态 map 的 peer index 是软件 rank；
2. map 中的 HCA index 必须对应编译时看到的 netdev 顺序；
3. `NCCL_IB_HCA` 可以筛选或重排可见设备，但不能可靠地修复已经编译进二进制的 peer map；
4. rank 表、Ring 邻接和 HCA map 必须作为一个整体版本化。

## 5. 检查仓库是否包含编译期拓扑假设

提交或使用二进制前，先检查源码/patch 是否硬编码了以下内容：

```bash
rg -n \
  'RING-ONLY|ncclRingDevOverride|NCCL_IB_HCA|crossNicRing|ringPrev|ringNext|map\[4\]\[4\]' \
  src patches scripts
```

应特别确认：

- ring 顺序是否硬编码为某一套节点排列；
- 非环邻居是否被过滤；
- peer-to-HCA map 是否为编译期常量；
- channel 奇偶是否决定 HCA 选择；
- `NCCL_IB_HCA` 的顺序是否被假定为固定；
- 是否存在未使用的 `PEER_HCA` 或类似变量。

仓库文档中的 `node01/node02/...` 只代表作者当时的逻辑标签，不能自动证明当前现场的设备 index 相同。

## 编译与校验建议

源码可以在开发机下载和打包，但应在目标 Linux ARM64、目标 CUDA 镜像内编译：

```bash
git clone --depth 1 --branch <NCCL_TAG> <NCCL_REPO> /src/nccl
cd /src/nccl
patch -p1 < /src/v5-full-chain.patch
# 按第 2～4 步生成的 map 修改 src/transport/net.cc
make -j"$(nproc)" src.build \
  CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
sha256sum build/lib/libnccl.so.2.30.7
```

编译不需要 GPU，但运行时需要目标 GPU、CUDA、RDMA 设备和正确的容器权限。必须记录：

- NCCL 源码 tag/commit；
- patch 版本；
- CUDA/Torch/vLLM 版本；
- 产物大小、SHA-256、MD5；
- map 对应的 rank 表和物理边表。

## 最小四 rank collective 验收

模型加载前，使用无模型的 PyTorch/NCCL all-reduce。每个节点只启动一个 rank，容器只挂载测试脚本和待测 NCCL 库：

```bash
docker run --rm \
  --entrypoint python3 \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 8g \
  --ulimit memlock=-1 \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/uverbs2 \
  --device /dev/infiniband/uverbs3 \
  -v <NCCL_LIB>:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2:ro \
  -v TP4_MINIMAL_NCCL_COLLECTIVE.py:/tmp/collective.py:ro \
  -e MASTER_ADDR=<RANK0_MANAGEMENT_IP> \
  -e MASTER_PORT=<CONTROL_PORT> \
  -e RANK=<0..3> \
  -e WORLD_SIZE=4 \
  -e NCCL_IB_GID_INDEX=<GID_INDEX> \
  -e NCCL_IB_HCA=<HCA_LIST> \
  -e NCCL_NET=IB \
  -e NCCL_SOCKET_IFNAME=<MANAGEMENT_IFACE> \
  -e GLOO_SOCKET_IFNAME=<MANAGEMENT_IFACE> \
  -e NCCL_DEBUG=INFO \
  <IMAGE> /tmp/collective.py
```

通过条件：

- 四个 rank 均完成初始化；
- 日志显示 `NET/IB`，不能回退到 `NET/Socket`；
- GID/HCA/peer 邻接与记录一致；
- 四个 rank 均打印 `TP4_NCCL_COLLECTIVE_PASS`；
- 没有 `ibv_modify_qp ... 110` 或 “no transport for peer” 错误。

只有 collective 通过后，才进入模型加载和 vLLM API smoke。测试结束应删除精确命名的临时容器，不执行无差别 `docker system prune`。

## 建议作者在仓库中补充的能力

可考虑将本说明转化为仓库的 preflight/diagnostic 改进：

1. 启动前打印 `rank → hostname → HCA/netdev → GID` 完整表；
2. 提供配置文件定义节点顺序和每条环边的 HCA pair；
3. 在编译或启动时校验 map 与 rank 数、环邻接是否一致；
4. 对非直连 HCA 连接给出“物理边/HCA map 不一致”的明确错误；
5. 将物理点到点 RDMA preflight 与 NCCL collective preflight 分开；
6. 避免未使用的 `PEER_HCA` 变量和无法生效的运行时重排暗示；
7. 提供 map 版本、源码 commit 和二进制 SHA-256 的发布记录。

## 本次现场得到的通用结论

- 物理链路可以全部正常，但 NCCL 仍因静态 map 错误而失败；
- 作者文档中的线序不能替代当前机器的点到点 RDMA 测试；
- 只交换 rank 或只交换 `NCCL_IB_HCA` 列表都不足以证明拓扑正确；
- 只有“物理边 + 软件 rank + 编译期 map + 最小 collective”四项同时通过，TP=4 才具备继续加载模型的条件。

## 匿名化验证实例

为便于 PR 讨论，可以只保留以下匿名信息：

```text
软件 rank：0=A，1=B，2=C，3=D
Ring：A ↔ B ↔ C ↔ D ↔ A
物理 HCA pair：AB=0/2，BC=1/3，CD=0/2，DA=1/3
```

在该 rank 表下，按现场 map 重编译的 NCCL `2.30.7` collective 通过；仓库原始 v5 二进制（SHA-256 `2138fc54…e1cd`）以及仅交换 rank/HCA 列表的变体均未通过。发布给作者时应同时附上源码 commit、patch 版本和二进制 SHA-256，避免只凭 tag 或文件名判断库是否相同。
