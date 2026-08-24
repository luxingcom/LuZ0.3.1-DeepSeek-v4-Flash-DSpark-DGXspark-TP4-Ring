"""routeA 最终基准：正确性(vs 官方 dequantize_to_dtype) + 性能(GPU GEMM)。"""
import torch, sys, time
torch.manual_seed(0); dev='cuda'
sys.path.insert(0,'/tmp')
import importlib, nvfp4_4w4a_mmaf as ma
importlib.reload(ma)
import vllm._custom_ops as co
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype, kE2M1ToFloat_handle
kE2M1ToFloat_handle.val=kE2M1ToFloat_handle.val.cuda()

E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float32)
def make_weights(K,N,scale,device):
    w=(torch.rand(K,N,device=device)*2-1)*scale
    wsr=w.abs().amax(dim=0).clamp(min=1e-9); wsb=wsr.view(N//128,128).amax(dim=1)
    exp=torch.floor(torch.log2(wsb.clamp(min=1e-30)/6.0))+127.0; exp=exp.clamp(0,255).to(torch.uint8)
    ws=exp.unsqueeze(0).repeat(K//32,1); wsf=torch.pow(2.0,ws.float()-127.0); wse=wsf.repeat_interleave(32,0).repeat_interleave(128,1)
    wn=w/wse; sgn=torch.sign(wn); wa=wn.abs(); pos=E2M1[:8].to(device)
    idx=(wa.unsqueeze(-1)-pos).abs().argmin(dim=-1); q=(sgn*pos[idx]).nan_to_num(0)
    mag=torch.tensor([0,1,2,3,4,5,6,7],device=device)
    mv=torch.where(q.abs().unsqueeze(-1)==pos.unsqueeze(0).unsqueeze(0),mag,torch.zeros_like(mag)).sum(dim=-1).to(torch.uint8)
    nib=(mv|((q<0).to(torch.uint8)*8)).to(torch.uint8)
    return (((nib[:,0::2]|(nib[:,1::2]<<4))&0xFF).contiguous()), ws.contiguous(), w

def official_ref(A, W_packed, W_scale):
    # 同一 W：适配层 dequant 出的 fp32
    W_fp32 = ma._dequant_w_our(W_packed, W_scale).t().contiguous()  # [N,K]
    gs=torch.tensor([1.0],dtype=torch.float32,device=dev)
    aq,asf=co.scaled_fp4_quant(A.half(), gs, True,'none')
    wq,wsf=co.scaled_fp4_quant(W_fp32.half(), gs, True,'none')
    a_d=dequantize_to_dtype(aq, asf, gs, torch.float32, 16, True)
    w_d=dequantize_to_dtype(wq, wsf, gs, torch.float32, 16, True)
    return a_d@w_d.t()

shapes=[(256,4096,4096),(512,2048,4096),(1024,4096,2048),(128,4096,4096),(256,8192,8192),(512,8192,8192),(1024,8192,4096),(256,4096,16384)]
print("=== routeA 适配层: (preprocess + GEMM) 正确性与性能 ===")
allok=True
for M,K,N in shapes:
    A=torch.randn(M,K,device=dev,dtype=torch.float32); Wp,Ws,_=make_weights(K,N,0.5,dev)
    ref=official_ref(A,Wp,Ws)
    impl=ma.RouteA()
    impl.preprocess_weights(Wp,Ws)   # 每层一次（测量外）
    out=impl(A,use_cached_w=True)
    torch.cuda.synchronize()
    rel=(out-ref).abs().sum().item()/(ref.abs().sum().item()+1e-9)
    for _ in range(5): impl(A,use_cached_w=True)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(30): impl(A,use_cached_w=True)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/30
    tf=2*M*K*N/dt/1e12
    ok=rel<0.02; allok=allok and ok
    print(f"M{M:4d} K{K:4d} N{N:5d}: rel={rel:.5f} [{'PASS' if ok else 'FAIL'}] | {dt*1e3:7.3f}ms | {tf:6.1f} TFLOPS")
print("\n== ALL PASS ==" if allok else "\n== SOME FAIL ==")
# preprocess 开销
M,K,N=256,4096,4096; A=torch.randn(M,K,device=dev); Wp,Ws,_=make_weights(K,N,0.5,dev)
impl=ma.RouteA(); t0=time.perf_counter(); impl.preprocess_weights(Wp,Ws); torch.cuda.synchronize()
print(f"W 预处理(每层一次): {(time.perf_counter()-t0)*1e3:.1f} ms")