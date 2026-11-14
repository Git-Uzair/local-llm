# Benchmarks, CUDA Buffer Math & Model Comparisons (RTX 4090 + 32 GB DDR5)

Empirical measurements, exact CUDA buffer breakdowns (`-lv 4` + `nvidia-smi`), and Dense vs. Mixture-of-Experts (MoE) scaling trade-offs on a **24 GB NVIDIA RTX 4090 + 32 GB DDR5-6000 Windows 11** workstation.

---

## 1. Reference Hardware Baseline

| Component | Specification | Practical Consequence for Local LLMs |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4090 (`Ada Lovelace`, SM 8.9) | **`1,008 GB/s`** GDDR6X bandwidth; native hardware FP8 (`NATIVE_FP8 = 1` FlashAttention) + fast MMQ/Tensor kernels. |
| **Total vs. Free VRAM** | `24,564 MiB` (`23.99 GiB`) total | Windows 11 WDDM + desktop compositor + browser typically consume `~1.0–2.0 GiB` (`22.5–23.4 GiB` free idle; `~23.7 GiB` free if monitors run off the CPU iGPU). |
| **CPU & System RAM** | 8C/16T X3D CPU + `32 GiB` DDR5-6000 (`31.10 GiB` usable) | `~60–70 GB/s` practical RAM bandwidth (**~15× slower** than 4090 GDDR6X). Use `8–12 GiB` for `--cache-ram` prompt caching—**never** spill dense model layers across PCIe. |
| **Runtime** | `llama.cpp v1.0.0` (`b12359`, `win-cuda-12.4-x64`) | Native Windows CUDA binary; avoid Vulkan on RTX 4090 (~50% slower prefill). |

---

## 2. Live `Qwen3.8-27B` + MTP Benchmarks on RTX 4090 (`v1.0.0` / `b12359`)

Tested with `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf` (`12.52 GiB` / `13.44 GB`, `64` main hybrid blocks + `1` embedded `Q8_0` MTP NextN block):

### 2.1 Configuration Sweep & Why `--fit off` + `-ub 512` Matters

| Run | Key Flags Changed | `nvidia-smi` Used / Total | Prefill (`tok/s`) | Decode (`tok/s`) | MTP Acceptance | Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1 (Broken default `--fit`)** | `--fit on -c 229376 -ub 2048 -ctk f8_e4m3` | `22,964 / 24,564 MiB` | `274.2` | **`12.8`** | `90.7%` | `--fit on` saw `3.70 GiB` dual compute buffers cross its `1024 MiB` margin and spilled dense layers to DDR5 RAM (**9.3× slowdown**). |
| **Run 2 (100% GPU, `q8_0` KV)** | `-ngl all -ngld all --fit off -c 196608 -ub 2048 -ctk q8_0` | `23,387 / 24,564 MiB` | — | **`119.6`** | `74.2%` (`1412/1902`) | Pinning `66/66` layers to `CUDA0` restored full GDDR6X + MTP speed immediately. |
| **Run 3 (Full `262K` Context)** | `-ngl all -ngld all --fit off -c 262144 -ub 512 -ctk f8_e4m3` | `23,799 / 24,564 MiB` | `1,905.6` | **`95.8–101.6`** | `75.5%` (`830/1099`) | Full `262,144` window works, but `<800 MiB` free VRAM slightly brushes WDDM shared memory on an active desktop. |
| **Run 4 (Production Default)** | **`-ngl all -ngld all --fit off -c 212992 -ub 512 -ctk f8_e4m3`** | **`23,053 / 24,564 MiB`** | **`2,381.1`** | **`119.5–129.9`** | **`75.6%–81.0%`** | **Optimal sweet spot:** `212,992` (`208K`) context, `1.5 GiB` free physical VRAM, `2,381 tok/s` prefill, `120+ tok/s` decode. |

### 2.2 Exact `CUDA0` Buffer Allocation Breakdown (`-lv 4` Verified)

For `Qwen3.8-27B` (`Q3_K_P`, `66/66` layers on `CUDA0`, `--spec-type draft-mtp --spec-draft-n-max 3`):

- **Model weights (`64` main + `1` MTP + `output`):** `12,286.92 MiB`
- **Gated DeltaNet recurrent state (`3` rollback snapshots for `n-max = 3`):** `598.50 MiB` (`+ 40.02 MiB` `h_nextn`)
- **Main + MTP Draft compute buffers (`-b 2048 -ub 512`):** `1,368.02 MiB` + `1,360.02 MiB` = **`2,728.04 MiB`** *(vs. `3,792.14 MiB` at `-ub 2048`)*
- **Per-token KV cache (`16` main full-attention layers + `1` MTP full-attention layer):**
  - Main `f8_e4m3` KV = `32.0 KiB/tok` (`6,656.00 MiB` @ `212,992` tok; `8,192.00 MiB` @ `262,144` tok)
  - Draft `q8_0` KV = `2.125 KiB/tok` (`442.00 MiB` @ `212,992` tok; `544.00 MiB` @ `262,144` tok)

**General VRAM Formula (`Qwen3.8-27B` w/ MTP on CUDA):**

$$\text{VRAM}_{\text{MiB}} \approx \text{Weights}_{\text{MiB}} + 639 + \text{Compute}_{\text{MiB}}(\text{ub}) + \text{Context} \times \left(\text{KV}_{\text{main}} + \text{KV}_{\text{draft}}\right)$$

---

## 3. Dense `Qwen3.8-27B` vs. MoE `35B-A3B` (`Qwen3.6` / `Ornith-1.5`)

When should you pick the dense `Qwen3.8-27B` setup in this repo versus a `35B-A3B` Mixture-of-Experts (`qwen35moe`, 35B total / 3B active)?

### 3.1 Single-Stream Hard Coding vs. Multi-Slot Subagent Concurrency

| Metric | **`Qwen3.8-27B` (`Q3_K_P` / `GSQ-RCO-IQ3_S` + MTP)** | **`Ornith-1.5-35B-A3B` / `Qwen3.6-35B-A3B` (`UD-IQ3_XXS` / `UD-IQ4_XS`)** |
| :--- | :--- | :--- |
| **Architecture** | Dense hybrid (`27B` active/tok, `16` full-attn layers) | MoE hybrid (`~3B` active/tok, `10` full-attn layers) |
| **Terminal-Bench 2.1 / SWE-bench Pro** | **`73.0` / `61.7`** (`Qwen3.8-27B` base) | `67.8` / `59.6` (`Ornith-1.5`) · `52.5` / `49.5` (`Qwen3.6`) |
| **KV Cost / Token (`q8_0` / `f8_e4m3`)** | `32–34 KiB/tok` | **`10.6 KiB/tok`** (~3× smaller per slot) |
| **1-Slot (`-np 1`) Coding Speed on 4090** | **`119.5–129.9 tok/s`** (with `draft-mtp`) | `76–85 tok/s` plain (`150–180 tok/s` on tuned Q4/MTP) |
| **2–4 Parallel Slots (`-np 2..4`)** | Drops to `~55–62 tok/s` shared (MTP is 1-slot) | Scales to **`119 tok/s` (`-np 2`)**, **`152 tok/s` (`-np 4`)**, **`178 tok/s` (`-np 8`)** |

### 3.2 Multi-Slot KV Thrashing Findings (When Running Concurrent Agents)

Measured across two simultaneous multi-step tool-calling sessions:

- **`-np 1` (single slot shared by 2 concurrent agents):** Every alternating request evicts the other agent's KV cache from VRAM. A workload requiring `42,053` prompt tokens alone was forced to re-prefill **`307,347` tokens** (`555 s` wall time).
- **`-np 2 --no-cache-idle-slots` (dedicated slot per agent):** Each agent stays pinned in its own slot (`46,618` prompt tokens, **`213 s` wall time — 2.6× faster**). Always pass `--no-cache-idle-slots` when `-np >= 2` so `llama-server` does not copy idle slot states to RAM between turns (cut mean first-token latency from `7.2 s` to `3.0 s` on RTX 4090).

### 3.3 Alternative Verified Hugging Face Checkpoints for 24 GB VRAM

| Use Case | Repository & Recommended File | Exact File Size | Zero-Spill Context on 24 GB Win11 |
| :--- | :--- | :--- | :--- |
| **Uncensored Hard Coding + MTP (Default)** | `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` → `...-Q3_K_P.gguf` | `12.52 GiB` | `212,992` (`208K`) |
| **Official Clean Hard Coding + Max Context** | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | `11.29 GiB` | `262,144` (`256K`) |
| **Uncensored Compact RCO + Full 256K** | `0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF` → `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_S-mtp.gguf` | `11.41 GiB` | `262,144` (`256K`) |
| **Parallel Subagents (`-np 2`) Fast MoE** | `unsloth/Qwen3.6-35B-A3B-GGUF` → `UD-IQ3_XXS.gguf` (`12.30 GiB`) or `UD-IQ4_XS.gguf` (`16.51 GiB`) | `12.30` / `16.51 GiB` | `2 × 131,072` (`2 × 128K`) |
| **High-Scoring Coding MoE (`-np 1..2`)** | `ornith-ai/Ornith-1.5-35B-A3B-GGUF` → `Ornith-1.5-35B-IQ3_M.gguf` / `Q4_K_M.gguf` | `15.4` / `20.22 GiB` | `2 × 131K` (`IQ3_M`) / `1 × 131K` (`Q4_K_M`) |
