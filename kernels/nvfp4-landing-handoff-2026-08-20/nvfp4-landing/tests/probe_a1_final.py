import torch, vllm._custom_ops as co
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
torch.manual_seed(0); dev='cuda'
M,K,N=512,4096,4096
A=torch.randn(M,K,device=dev)*0.5
W=torch.randn(N,K,device=dev)*0.5
gs=torch.tensor([1.0],dtype=torch.float32,device=dev)

# official quant (block 16)
def quant_o(x):
    fp4, sf = ref_nvfp4_quant(x.float(), gs, 16)   # fp4 [m,n] fp32 values clamped; sf [m, n/16] float
    return fp4, sf
# packed: cast fp4 values to codes (0/0.5/1/1.5/2/3/4/6 -> mag 0..7, sign)
MAG=[0,1,2,3,4,5,6,7]; P=[0.,0.5,1.,1.5,2.,3.,4.,6.]
def pack(x):  # x [m,n] fp32 values in {-6..6} set -> [m,n//2] uint8
    m,n=x.shape
    s=torch.sign(x); a=x.abs()
    idx=torch.argmin((a.unsqueeze(-1)-torch.tensor(P,device=x.device)).abs(),dim=-1).long()
    nib=t_idx=idx.to(torch.uint8) | ((s<0).to(torch.uint8)<<3)
    lo=nib[:,0::2]; hi=nib[:,1::2]
    return ((lo|(hi<<4))&0xFF).contiguous()

a_fp4,a_sf=quant_o(A)
w_fp4,w_sf=quant_o(W)   # but W should be [N,K] -> w_sf [N,K/16]
a_q=pack(a_fp4); w_q=pack(w_fp4)
# sf to e4m3 swizzle:
def to_e4m3_sw(sf):  # sf[m, K/16] float -> [m_padded, K/16] e4m3 swizzled
    return swizzle_blockscale(sf.to(torch.float8_e4m3fn))
a_sf_sw=to_e4m3_sw(a_sf)
w_sf_sw=to_e4m3_sw(w_sf)

out=co.cutlass_scaled_fp4_mm(a_q,w_q,a_sf_sw,w_sf_sw, torch.tensor([1.0],dtype=torch.float32,device=dev), torch.bfloat16).float()

# reference: official dequant math: a_dq = a_fp4 * (sf/gs); w_dq = w_fp4 * (wsf/gs)
a_dq=(a_fp4.reshape(M,K//16,16) * (a_sf.unsqueeze(-1)/gs.item())).reshape(M,K)
w_dq=(w_fp4.reshape(N,K//16,16) * (w_sf.unsqueeze(-1)/gs.item())).reshape(N,K)
ref=a_dq @ w_dq.t()
rel=(out-ref).abs().sum().item()/(ref.abs().sum().item()+1e-9)
print(f"CUTLASS vs official-dequant-ref: rel={rel:.4f}")
print(f"  cut mag {out.abs().mean().item():.4f} ref mag {ref.abs().mean().item():.4f}")
# also raw fp32
relf=(out-A@W).abs().sum().item()/((A@W).abs().sum().item()+1e-9)
print(f"  CUTLASS vs raw fp32: rel={relf:.4f}")