"""routeA(原生FP4) vs v15(bf16 MMA) 同 shape 性能对比 —— 证明原生路径价值。"""
import torch, sys, time
torch.manual_seed(0); dev='cuda'
sys.path.insert(0,'/tmp'); import nvfp4_4w4a_mmaf as ma
M,K,N=512,4096,4096
A=torch.randn(M,K,device=dev)*0.5
Wp=torch.randint(0,16,(K,N//2),dtype=torch.uint8,device=dev)
Ws=torch.full((K//32,N//128),127,dtype=torch.uint8,device=dev)

impl=ma.RouteA(); impl.preprocess_weights(Wp,Ws)
for _ in range(5): impl(A,use_cached_w=True)
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(200): impl(A,use_cached_w=True)
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/200
tf=2*M*K*N/dt/1e12
print(f"routeA 原生FP4  512x4096x4096: {dt*1e3:.3f}ms = {tf:.1f} TFLOPS")

# v15 bf16 baseline: raw torch bf16 matmul (approx v15 kernel which is bf16 MMA)
Ah=A.half(); Wb=(A @ Wp.float())[0:0].half() if False else torch.randn(N,K,device=dev).half()
# v15 = bf16 MMA path; approximate with bf16 matmul as upper-bound proxy
for _ in range(5): C=Ah@Wb
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(200): C=Ah@Wb
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/200
tf2=2*M*K*N/dt/1e12
print(f"bf16 matmul(近似v15) 512x4096x4096: {dt*1e3:.3f}ms = {tf2:.1f} TFLOPS")
print(f"加速比(routeA/approx_v15): {tf/tf2:.2f}x")