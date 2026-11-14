# Local Coding LLM on Windows 11 (RTX 4090 24 GB + 32 GB RAM)

Reproducible setup, tuning scripts, and benchmark-backed parameter guide for running a **208K–256K context coding & tool-calling LLM** at **119–130 tok/s generation** and **2,380+ tok/s prompt prefill** 100% inside a single **24 GB NVIDIA RTX 4090** on native Windows 11—with one-command integration into the **Kilo CLI** coding agent harness.

---

## Measured Reference Performance

Benchmarked live on Windows 11 (`NVIDIA GeForce RTX 4090 24,564 MiB` WDDM, `32 GiB DDR5-6000`, `llama.cpp v1.0.0 / b12359` `win-cuda-12.4-x64`):

| Workload | Measured Result | Notes |
| :--- | :--- | :--- |
| **Active Context Window** | **`212,992` tokens (`208K`)** | `66/66` layers (`64` main + `1` MTP + `output`) 100% in GDDR6X (`~1.5 GiB` physical VRAM still free for desktop/browser). |
| **Codebase Prompt Prefill (`7,506` tok)** | **`2,381.1 tok/s`** | `3.15 s` cold prefill with Flash Attention + native Ada SM 8.9 `f8_e4m3` KV cache. |
| **Sustained Coding Generation (`3,072` tok)** | **`119.5 tok/s`** | `75.6%` Multi-Token Prediction (`draft-mtp`) acceptance across concurrent Python + pytest code. |
| **Agentic Tool Calls (`JSON` schema)** | **`129.9 tok/s`** | `81.0%` MTP draft acceptance (`0.74 s` total turn latency). |
| **Multi-Turn Tool Follow-Up** | **`124.6 tok/s` (`100%` KV prefix hit)** | Preserved `<think>` history keeps prompt prefix stable across tool loops (`0.53 s` turn latency). |

---

## Hardware & Software Prerequisites

- **GPU:** NVIDIA RTX 4090 (`24 GB` VRAM, Ada Lovelace SM 8.9) or any 24 GB+ NVIDIA GPU (`RTX 3090` works; use `-KvType q8_0` on Ampere SM 8.6 since native FP8 requires SM 8.9+).
- **System RAM:** `32 GB` DDR5/DDR4 (`~8 GB` reserved for `llama-server` `--cache-ram`).
- **OS & Driver:** Windows 11 64-bit with a current NVIDIA Game Ready / Studio driver (`CUDA 12.4+` UMD; no full CUDA Toolkit installation required—runtime DLLs are downloaded automatically).
- **Tools:** Windows PowerShell 5.1 or PowerShell 7 (`pwsh`), plus Python 3.10+ (only needed if running `scripts/configure-kilo.py`). Optional fast downloaders (`gh` and `uv` / `hf`) are used automatically if installed, with built-in HTTPS fallbacks otherwise.

---

## 3-Step Quickstart

Clone the repository and run from PowerShell:

```powershell
git clone https://github.com/Git-Uzair/local-llm.git
cd local-llm

# 1. Download llama.cpp v1.0.0 (CUDA 12.4) + GGUF model (~12.5 GiB) and verify SHA-256
.\scripts\setup.ps1

# 2. Register the local endpoint in ~/.config/kilo/kilo.json(c) (optional, for Kilo CLI users)
python .\scripts\configure-kilo.py

# 3. Start the tuned inference server on http://127.0.0.1:10088/v1
.\scripts\start-server.ps1
```

Then start Kilo CLI directly against your local GPU:

```powershell
kilo -m local-llama/qwen3.8-27b-coder
```

*(Or switch inside an open Kilo TUI session via `/models` or `Ctrl+X` then `m`.)*

### Storing Binaries & Weights on a Separate Drive

By default, `setup.ps1` and `start-server.ps1` use `<repo>\runtime\llama-cpp` and `<repo>\models` (both `.gitignore`d). To keep large files on another disk without modifying tracked scripts, set environment variables or pass CLI flags:

```powershell
$env:LOCAL_LLM_RUNTIME_DIR = "D:\LLM\Software\llama-cpp"
$env:LOCAL_LLM_MODELS_DIR  = "D:\LLM\Models"

.\scripts\setup.ps1
.\scripts\start-server.ps1
```

To install the official non-abliterated IST Austria DASLab build (`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, `11.29 GiB`) instead of the default uncensored build:

```powershell
.\scripts\setup.ps1 -Profile clean
```

---

## Why Every Architectural & Setup Choice Was Made

### 1. Inference Engine: Why Native Windows `llama-server.exe` (`v1.0.0` CUDA) Wins

| Engine on Windows 11 | Verdict on 1× RTX 4090 + 32 GB RAM | Technical Reason |
| :--- | :--- | :--- |
| **Native `llama-server.exe` (`v1.0.0` CUDA 12.4)** | **Selected (#1)** | Zero VM RAM/VRAM overhead; native `qwen35` fused Gated DeltaNet kernels; native `--spec-type draft-mtp` speculative decoding (`+60–100%` decode tok/s); native Ada SM 8.9 `f8_e4m3` FlashAttention KV cache; reference-counted recurrent checkpoints; 15–21% faster single-stream decode than ExLlamaV3 at matched weight size. |
| **WSL2 + Docker + vLLM / SGLang** | Rejected | Built for 16–64 concurrent users, not 1–2 local streams. WSL2's Hyper-V VM carves GiBs out of tight **32 GB system RAM**, adds ~1 GiB CUDA/WSL overhead, pre-allocates VRAM statically, and vLLM's GGUF loader remains experimental. |
| **ExLlamaV3 + TabbyAPI** | Niche Runner-Up | 1.3–2.2× faster *cold* prompt prefill on safetensors/EXL3, ~18% slower non-MTP decode, and cannot cleanly convert pre-quantized GGUFs without pulling 54 GB `BF16` source checkpoints. |
| **Ollama / LM Studio GUI** | Rejected for Peak Perf | Ollama hides `--spec-type draft-mtp` and draft KV quantization (`-ctkd`/`-ctvd`) and breaks on long Windows GGUF filenames. LM Studio's Electron UI consumes ~300–600 MiB of WDDM VRAM needed for `200K+` context. |

### 2. Model Architecture: Why `Qwen3.8-27B` (`qwen35` Hybrid) + Smart Per-Tensor Quantization

- **Hybrid Linear + Full Attention (`64` layers = `48` Gated DeltaNet + `16` Full Attention):**
  - Standard dense 27B–32B transformers grow KV cache across all 64 layers (`~256 KiB/token` at `f16` → **32 GiB** for 128K context alone, impossible in 24 GB VRAM).
  - In `Qwen3.8-27B`, **48 of 64 layers are Gated DeltaNet** using a constant-size recurrent state (`149.6 MiB` total regardless of prompt length). Only **16 layers** (`4 KV heads × 256 dim`) allocate per-token KV cache: **`32 KiB/token` at `f8_e4m3`** (`6.50 GiB` for `208K` tokens, `8.00 GiB` for `256K` tokens).
- **Why Smart `3.5–3.9 bpw` MTP Quants (`11.3–12.5 GiB`) Beat `17–19 GiB` `Q4_K_M` / `MAX` Quants on a 24 GB Card:**
  1. **Memory-bandwidth physics (`1,008 GB/s`):** Dense token generation reads every model weight per forward pass. A `12.52 GiB` file moves **27% fewer weight bytes per step** than a `17.23 GiB` file (`~60 tok/s` plain vs `~43 tok/s`, scaling to **`119–130 tok/s`** with MTP).
  2. **Avoiding the `output.weight` BF16 trap:** `Qwen3.8-27B` has a `248,320 × 5,120` vocabulary head (`1.27B` parameters). "MAX" GGUFs that leave `output.weight` in 16-bit waste **`2.37 GiB` of VRAM** and lose **10–16% decode tok/s** on a single tensor.
  3. **Zero-spill 200K+ context:** Keeping weights at `11.3–12.5 GiB` leaves `~9.5–10.5 GiB` of VRAM for `208K–256K` KV cache + dual MTP compute buffers with **zero PCIe spill**.
- **Two Verified Profiles Included in `setup.ps1`:**
  - **`-Profile uncensored` (Default):** `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` (`Q3_K_P`, `12.52 GiB`, `3.93 bpw`). Direct abliteration of stock `Qwen/Qwen3.8-27B` (`0/465` refusals, zero roleplay Frankenmerge drift, intact MTP head achieving `75–92%` draft acceptance).
  - **`-Profile clean`:** `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` (`IQ3_S-mtp`, `11.29 GiB`, `3.50 bpw`). First-party IST Austria DASLab Riemannian Constrained Optimization quant matching 54 GB `BF16` **100.0% identically** on LiveCodeBench v6 (`85.71`) and AIME25 (`100.00`).
- **When to Prefer an MoE (`35B-A3B`) Instead:** MTP speculative decoding only accelerates `-np 1` (one active request at a time). If you run **2–4 parallel subagents simultaneously (`-np 2` / `-np 4`)**, a 3B-active MoE (`ornith-ai/Ornith-1.5-35B-A3B-GGUF` or `unsloth/Qwen3.6-35B-A3B-GGUF` at `UD-IQ3_XXS` / `UD-IQ4_XS`) delivers `120–180 tok/s` combined throughput across parallel slots (see [`docs/benchmarks-and-vram-math.md`](docs/benchmarks-and-vram-math.md)).

### 3. Windows 11 Host-Level Optimizations for 24 GB GPUs

1. **Offload Monitors/Browsers to Your CPU's iGPU (Reclaims `~1.2–1.8 GiB` VRAM):** On CPUs with integrated graphics (e.g., AMD Ryzen 7000/9000 `Radeon Graphics` or Intel non-`F` CPUs), plugging displays into the motherboard port—or assigning browsers/Electron apps to the *Power Saving* iGPU under *Settings → System → Display → Graphics* and disabling *NVIDIA Overlay*—frees enough 4090 VRAM to raise `-ContextSize` from `212992` (`208K`) to the full `262144` (`256K`).
2. **Prevent Silent WDDM Shared-Memory Paging:** Under *NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy*, select **`Prefer No Sysmem Fallback`**. By default on Windows WDDM, exceeding physical VRAM by even 300 MiB silently pages CUDA tensors over PCIe (`~65 GB/s` DDR5 vs `1,008 GB/s` GDDR6X), cutting generation speed by 50–90% while `/health` still reports `200 OK`.

---

## Complete `llama-server` Parameter & Trade-Off Reference

Every flag in `scripts/start-server.ps1` was validated by live A/B testing. Here is what each parameter controls and how changing it affects speed, VRAM, and output quality:

### 1. GPU Offloading & Memory Fitting (`-ngl all -ngld all --fit off`)

- **Why `--fit off` + `-ngl all -ngld all` instead of `llama-server`'s default `--fit on`:**
  - In live benchmarking on a 24 GB RTX 4090, leaving `--fit on` at `229K` context caused `llama-server` to see the dual main + MTP compute buffers (`~2.7–3.7 GiB`) cross its default `--fit-target 1024` MiB reserve and **silently offload dense layers to CPU RAM**, collapsing generation speed from **`119.5 tok/s` down to `12.8 tok/s` (a 9.3× slowdown)**.
  - Explicitly passing `-ngl all` (main model), `-ngld all` (MTP draft head), and `--fit off` pins all `66/66` layers in GDDR6X VRAM.
- **What happens if you change it:** If you increase `-ContextSize` or open VRAM-heavy 3D apps beyond `24,564 MiB`, `--fit off` (paired with `Prefer No Sysmem Fallback`) fails fast at startup with an explicit CUDA out-of-memory error rather than running 10× slower in secret—simply lower `-ContextSize` (e.g., `-ContextSize 163840`).

### 2. Batch & Micro-Batch Sizing (`-b 2048 -ub 512`)

- **`-b 2048` (logical batch):** Maximum prompt chunk processed per scheduler step.
- **`-ub 512` (physical CUDA micro-batch) — Critical VRAM lever when MTP is enabled:**
  - With `--spec-type draft-mtp`, `llama-server` allocates **two** separate CUDA compute buffers (one for the 64-layer target model, one for the MTP draft context):
    - At `-ub 2048`: `1,936 MiB` (main) + `1,856 MiB` (draft) = **`3,792 MiB` (`3.70 GiB`)** of compute buffers.
    - At **`-ub 512`**: `1,368 MiB` (main) + `1,360 MiB` (draft) = **`2,728 MiB` (`2.66 GiB`)**, **saving `1,064 MiB` (`>1.0 GiB`) of VRAM**—equivalent to `+34,000` extra context tokens—while still hitting **`2,381 tok/s` prompt prefill**.
- **What happens if you change it:** Raising `-ub` to `2048` adds ~5–12% prefill throughput on non-MTP or MoE setups with spare VRAM, at the cost of `+1.04 GiB` VRAM when MTP is active. Lowering `-ub` below `512` (`256`) saves ~300 MiB more VRAM but noticeably throttles long-prompt prefill.

### 3. Context Window & Slot Division (`-c 212992 -np 1`)

- **`-c` sets total KV cells divided across `-np` parallel slots (`per_slot_tokens = -c / -np`).**
- **VRAM scaling on `Qwen3.8-27B` (`-np 1` with `f8_e4m3` main KV + `q8_0` draft KV):**
  - `-c 131072` (`128K`): `4,096 MiB` main + `272 MiB` draft → `~20.0 GiB` total VRAM (`~4.0 GiB` free for heavy desktop/games or `--mmproj` vision).
  - **`-c 212992` (`208K`, Default):** `6,656 MiB` main + `442 MiB` draft → **`23,053 MiB` (`22.5 GiB`)** total (`~1.5 GiB` free WDDM margin, **`119.5 tok/s`**).
  - `-c 262144` (`256K` max native): `8,192 MiB` main + `544 MiB` draft → `23,799 MiB` (`765 MiB` free; brushes WDDM on an active 4090 desktop at `95.8 tok/s`, or runs at full `~119 tok/s` when desktop displays run on the CPU iGPU).
- **What happens if you raise `-np` (`-ParallelSlots 2`):**
  1. Each slot only gets half of `-c` (`212,992 / 2 = 106,496` tokens per slot).
  2. MTP speculative decoding does not speed up multi-slot continuous batching (`start-server.ps1` automatically disables `draft-mtp` and enables `--no-cache-idle-slots` when `-ParallelSlots -ge 2`), so single-stream decode returns to dense baseline (`~55–62 tok/s`).

### 4. Flash Attention & KV Cache Quantization (`-fa on -ctk f8_e4m3 -ctv f8_e4m3 -ctkd q8_0 -ctvd q8_0`)

- **`-fa on`:** Mandatory for quantized V caches (`-ctv`), faster long-context attention, and lower compute buffer memory.
- **KV cache precision trade-offs (`Qwen3.8-27B`, 16 full-attention layers):**

  | `-KvType` (`-ctk` / `-ctv`) | Bytes / Token | VRAM for `208K` Context | Hardware Support & Quality Impact |
  | :--- | :--- | :--- | :--- |
  | **`f8_e4m3` (Default)** | **`32.0 KiB`** | **`6.50 GiB`** | Native hardware FP8 FlashAttention on Ada Lovelace (`SM 8.9`, RTX 4090) & newer; near-lossless vs `f16` at **50% VRAM cost**. |
  | **`q8_0`** | **`34.0 KiB`** | **`6.91 GiB`** | Universal fallback for Ampere (`RTX 3090`, `SM 8.6`) where native FP8 is absent; virtually indistinguishable from `f16`. |
  | **`q4_0`** | **`18.0 KiB`** | **`3.66 GiB`** | Saves ~3 GiB VRAM (fits `262K` context even alongside `16 GiB` `Q4_K_M` weights), with slight needle-in-a-haystack precision loss on exact identifier recall at >128K depth. |
  | **`f16`** | **`64.0 KiB`** | **`13.00 GiB`** | Unquantized baseline; overflows 24 GB VRAM above `~112K` context with a 12.5 GiB model. |

### 5. Multi-Token Prediction Speculative Decoding (`--spec-type draft-mtp --spec-draft-n-max 3`)

- **How it works:** `Qwen3.8-27B` was trained with an architectural `nextn` Multi-Token Prediction head (`+0.42 GiB` in the GGUF). On each step, the lightweight MTP head drafts `N` future tokens (`--spec-draft-n-max 3`) and the 27B model verifies all `N + 1` tokens in a single batched forward pass—turning memory-bandwidth-bound matrix-vector decode into compute-efficient small-batch matmuls (`~58 tok/s → 119.5–129.9 tok/s`).
- **Parameters that make or break MTP:**
  - **`--spec-draft-n-max` (`-DraftMax`):** `2` or `3` is optimal on RTX 4090 for code/JSON (`75–85%` acceptance). Above `4`, rejected tail drafts waste verification compute and recurrent state rollback bandwidth (`149.6 MiB × N`). For high-entropy prose where acceptance falls below ~55%, drop `-DraftMax` to `2` or `0`.
  - **`--repeat-penalty 1.0 --presence-penalty 0.0` & `--temp <= 1.0`:** **Do not raise repeat/presence penalty with MTP active.** Penalty samplers shift target token probabilities away from the MTP head's greedy/low-temp draft predictions, crashing acceptance below 45% and making MTP slower than plain decoding.

### 6. Reasoning Effort & Preserved Thinking (`--reasoning-effort medium --reasoning-preserve`)

- **How `Qwen3.8-27B` implements `reasoning_effort`:** Its Jinja template accepts **only `'low'`, `'medium'`, or `'xhigh'`** (passing `'high'` throws a Jinja template error; omitting the flag defaults to `'xhigh'`). Rather than cutting off tokens mid-thought, it injects a trained RL control instruction at the start of `<|im_start|>system`:
  - **`low`:** ~30–250 `<think>` tokens (`~0.3–2 s` wait before code/tool execution).
  - **`medium` (Server Default):** Neutral (`''`)—scales naturally from ~80 tokens (`0.7 s`) on routine `read_file` tool calls to ~1,500 tokens (`12 s`) on complex algorithms.
  - **`xhigh`:** ~1,000–8,000+ `<think>` tokens (`~8–65 s` of assumption validation and alternative checking; best for race conditions and multi-file architecture).
- **Overriding dynamically from Kilo CLI:** Selecting variant `low`, `medium`, or `xhigh` in Kilo CLI sends per-request `chat_template_kwargs.reasoning_effort`, overriding `--reasoning-effort medium` without restarting `llama-server`.
  - **KV Cache Caveat:** Because `reasoning_instructions` is injected around **token ~4** of the system prompt, switching variants *mid-conversation* changes the prompt prefix and triggers a one-time KV re-prefill (`~2,380 tok/s`). Pick your desired variant at the start of a task for 100% cache hits across every tool step.
- **`--reasoning-format deepseek --reasoning-preserve`:** Extracts `<think>...</think>` into `message.reasoning_content` and retains historical reasoning in the rendered prompt so multi-step tool loops match the slot's cached token sequence verbatim.
- **`-n 65536`:** Caps per-turn `max_tokens` (`<think>` + final code/tool call combined) high enough that `xhigh` turns never truncate mid-thought before emitting `<tool_call>`.

### 7. Host RAM Prompt Cache (`--cache-ram 8192 --ctx-checkpoints 8`)

- Reserves up to **8 GiB** of your 32 GB system RAM to hold evicted session KV prefixes and reference-counted `149.6 MiB` Gated DeltaNet recurrent checkpoints. Switching between multiple coding sessions restores cached prefixes from DDR5 RAM in seconds rather than re-prefilling 100K+ tokens, while leaving ~23 GiB of system RAM free for Windows 11, browsers, and IDEs.

---

## Kilo CLI Configuration Details

Running `python .\scripts\configure-kilo.py` merges the following provider entry into `~/.config/kilo/kilo.json` (or `kilo.jsonc`, creating a timestamped `.bak-*` backup first and automatically appending `"local-llama"` to `"enabled_providers"` if an allowlist is present):

```json
{
  "provider": {
    "local-llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local GPU (llama.cpp)",
      "options": {
        "baseURL": "http://127.0.0.1:10088/v1",
        "apiKey": "local-dev-key",
        "timeout": 600000,
        "kiloOpenAICompatibleToolCallContent": "empty-string"
      },
      "models": {
        "qwen3.8-27b-coder": {
          "name": "Qwen 3.8 27B Coder (208K MTP)",
          "tool_call": true,
          "reasoning": true,
          "temperature": true,
          "limit": {
            "context": 212992,
            "output": 65536
          },
          "variants": {
            "low": { "chat_template_kwargs": { "reasoning_effort": "low", "preserve_thinking": true } },
            "medium": { "chat_template_kwargs": { "reasoning_effort": "medium", "preserve_thinking": true } },
            "xhigh": { "chat_template_kwargs": { "reasoning_effort": "xhigh", "preserve_thinking": true } }
          }
        }
      }
    }
  }
}
```

**Why `"kiloOpenAICompatibleToolCallContent": "empty-string"` is mandatory:** By default, the OpenAI SDK sends `"content": null` on assistant messages that contain `tool_calls`. `llama-server v1.0.0` enforces string-typed `content` and returns HTTP `500` (`[json.exception.type_error.302] type must be string, but is null`) on Turn 2 without `"empty-string"`.

---

## Repository Layout

- `scripts/setup.ps1` — Downloads `llama.cpp v1.0.0` (`b12359` CUDA `12.4`) + model GGUF and verifies SHA-256 digests.
- `scripts/start-server.ps1` — Hardware-calibrated `llama-server.exe` launcher with CLI flags for context, KV type, slots, and MTP.
- `scripts/configure-kilo.py` — Idempotent, JSONC-safe `~/.config/kilo` updater (`--dry-run` supported).
- [`docs/benchmarks-and-vram-math.md`](docs/benchmarks-and-vram-math.md) — Detailed CUDA buffer measurements, VRAM formulas, and Dense (`Qwen3.8-27B`) vs. MoE (`Qwen3.6-35B-A3B` / `Ornith-1.5-35B-A3B`) multi-slot benchmarks.
