import time
import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_func as fa_hip
from flash_attn.flash_attn_interface import flash_attn_func as fa_triton
import csv

torch.set_grad_enabled(False)

# (qk_rope_dim, kv_rank, v_dim, num_q_head)
PARAMS = {
    'MLA_8B': (64, 160, 128, 32),
    'DeepSeekV2/V3': (64, 512, 128, 128),
    'KIMI': (64, 512, 128, 64),
}

# ---------------------------------------------------------------------------
# 1. helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def make_inputs(batch_size, qk_rope_dim, kv_rank, v_dim, num_q_head, num_kv_head, q_seq_len, kv_seq_len, device="cuda", dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    q = torch.randn((batch_size, num_q_head, q_seq_len, kv_rank+qk_rope_dim), dtype=dtype, device=device)
    kv_cache = torch.randn((batch_size, num_kv_head, kv_seq_len, kv_rank+qk_rope_dim), dtype=dtype, device=device)
    k = kv_cache
    v = kv_cache[..., :kv_rank]
    return q, k, v

def ref_mqa(q, k, v, scale):
    k_repeat = k.repeat(1, q.shape[1], 1, 1).contiguous()
    v_repeat = v.repeat(1, q.shape[1], 1, 1).contiguous()
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = attn_scores.softmax(dim=-1)
    return torch.matmul(attn_weights, v)

def our_mqa(q, k, v, scale):
    scores = torch.einsum("bshc,btc->bsht", q, k) * scale
    scores = scores.softmax(dim=-1)
    return torch.einsum("bsht,btc->bshc", scores, v)

def flash_attn_only(q, k, v, scale):
    return flash_attn_func(
        q, k, v,
        softmax_scale = scale,
        causal = False
    )

def flash_attn_fa_hip(q, k, v, scale):
    return fa_hip(
        q, k, v,
        softmax_scale = scale,
        causal = False
    )

def flash_attn_fa_triton(q, k, v, scale):
    return fa_triton(
        q, k, v,
        softmax_scale = scale,
        causal = False
    )

def sdpa_only(q, k, v, scale):
    out = F.scaled_dot_product_attention(
        q, k, v,
        scale = scale,
        is_causal = False,
    )
    return out

@torch.inference_mode()
def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

@torch.inference_mode()
def test_mla(bs, model, kv_seq_len, device="cuda", dtype=torch.bfloat16, seed=42):
    
    qk_rope_dim, kv_rank, v_dim, num_q_head = PARAMS[model]
    num_kv_head, q_seq_len = 1, 1
    
    # ----------- Create inputs ---------------------
    torch.manual_seed(seed)
    q = 0.02 * torch.randn((bs, num_q_head, q_seq_len, kv_rank+qk_rope_dim), dtype=dtype, device=device)
    k = 0.02 * torch.randn((bs, num_kv_head, kv_seq_len, kv_rank+qk_rope_dim), dtype=dtype, device=device)
    v = 0.02 * torch.randn((bs, num_kv_head, kv_seq_len, kv_rank), dtype=dtype, device=device)
    scale =  (qk_rope_dim * 2) ** (-0.5)
    
    # ----------- Prepare “ready” tensors for each impl ---------------------
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()

    q_ours = q.clone().permute(0, 2, 1, 3).contiguous()
    k_ours = k.clone().permute(0, 2, 1, 3).contiguous().squeeze(2)
    v_ours = v.clone().permute(0, 2, 1, 3).contiguous().squeeze(2)

    q_flash = q.clone().permute(0, 2, 1, 3).contiguous()
    k_flash = k.clone().permute(0, 2, 1, 3).contiguous()
    v_flash = F.pad(v.clone().permute(0, 2, 1, 3).contiguous(), [0, qk_rope_dim])

    q_sdpa = q.clone()
    k_sdpa = k.clone()
    v_sdpa = v.clone()
    
    try:
        # ----------- Correctness ---------------------------------------------
        # ref   = ref_mqa(q_ref, k_ref, v_ref, scale)
        # out_ours = our_mqa(q_ours, k_ours, v_ours, scale).permute(0, 2, 1, 3).contiguous()
        # out_f = flash_attn_only(q_flash, k_flash, v_flash, scale)[...,:kv_rank].permute(0, 2, 1, 3).contiguous()    
        # out_hip = flash_attn_fa_hip(q_flash, k_flash, v_flash, scale)[...,:kv_rank].permute(0, 2, 1, 3).contiguous()
        # out_triton = flash_attn_fa_triton(q_flash, k_flash, v_flash, scale)[...,:kv_rank].permute(0, 2, 1, 3).contiguous()
        # out_s = sdpa_only(q_sdpa, k_sdpa, v_sdpa, scale)
        
        # print(f"\nModel {model}, bs: {bs}, kv_seq_len: {kv_seq_len}, dtype: {dtype}")
        # for name, out in [("ours", out_ours), ("flash-attn", out_f), ("sdpa", out_s)]:
        #     ok = torch.allclose(ref, out, rtol=1e-4, atol=1e-4)
        #     print(f"{name:10s} match: {ok}")
        
        # ----------- Latency ----------------------------------------------------
        print(f"\nAverage forward latency (ms)")
        t_ref = benchmark(lambda: ref_mqa(q_ref, k_ref, v_ref, scale))
        print(f"  ref_mqa                       : {t_ref :7.3f}")
        t_ours = benchmark(lambda: our_mqa(q_ours, k_ours, v_ours, scale))
        print(f"  ours                          : {t_ours:7.3f}")
        t_fattn = benchmark(lambda: flash_attn_only(q_flash, k_flash, v_flash, scale))
        print(f"  flash_attn_func (default)     : {t_fattn:7.3f}")
        t_fattn_hip = benchmark(lambda: flash_attn_fa_hip(q_flash, k_flash, v_flash, scale))
        print(f"  flash_attn_func_hip           : {t_fattn_hip:7.3f}")
        t_fattn_triton = benchmark(lambda: flash_attn_fa_triton(q_flash, k_flash, v_flash, scale))
        print(f"  flash_attn_func_triton        : {t_fattn_triton:7.3f}")
        t_sdpa  = benchmark(lambda: sdpa_only(q_sdpa, k_sdpa, v_sdpa, scale))
        print(f"  SDPA (F.scaled_dot_product..) : {t_sdpa :7.3f}")

        return [
            model, bs, kv_seq_len, str(dtype),
            t_ref, t_ours, t_fattn, t_fattn_hip, t_fattn_triton, t_sdpa
        ]
            
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error occurred: {e}")
        return f"An unexpected error occurred: {e}"



# ---------------------------------------------------------------------------
# 2. main -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    results = []
    for model in ['MLA_8B', 'DeepSeekV2/V3', 'KIMI']:
        for bs in [1]:
            for kv_seq_len in [8192]:
                for dtype in [torch.bfloat16]:

                    res = test_mla(
                        bs, 
                        model,
                        kv_seq_len, 
                        device="cuda", 
                        dtype=dtype
                        )
                    if res: 
                        results.append(res)
    with open("MI300_micro_benchmarks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Batch Size", "KV Seq Len", "Dtype",
            "Ref MQA (ms)", "Ours (ms)", "Flash Attn (default) (ms)",
            "Flash Attn HIP (ms)", "Flash Attn Triton (ms)",
            "SDPA (ms)"
        ])
        writer.writerows(results)
                    


if __name__ == "__main__":
    main()
