"""routeA 大 shape 极限性能 + GEMM-only vs 全算子分离。"""
import torch, sys, time, vllm._custom_ops as co
torch.manual_seed(0); dev='cuda'
sys.path.insert(0,'/tmp'); import nvfp4_4w4a_mmaf as ma

print("=== routeA 大 M/K 原生 FP4 极限性能 ===")
for M,K,N in [(2048,4096,4096),(4096,4096,4096),(8192,4096,4096),(2048,8192,8192),(4096,8192,8192),(1024,12288,12288),(2048,4096,12288)]:
    A=torch.randn(M,K,device=dev)*0.5
    Wp=torch.randint(0,16,(K,N//2),dtype=torch.uint8,device=dev)
    Ws=torch.full((K//32,N//128),127,dtype=torch.uint8,device=dev)
    impl=ma.RouteA(); impl.preprocess_weights(Wp,Ws)
    for _ in range(3): impl(A,use_cached_w=True)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(10): impl(A,use_cached_w=True)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/10
    tf=2*M*K*N/dt/1e12
    print(f"M{M:5d} K{K:5d} N{N:5d}: {dt*1e3:7.3f}ms | {tf:6.1f} TFLOPS")
    torch.cuda.empty_cache()