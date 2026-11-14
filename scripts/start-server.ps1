param(
    [string]$RuntimeDir = '',
    [string]$ModelsDir = '',
    [string]$ModelPath = '',
    [string]$Alias = 'qwen3.8-27b-coder',
    [int]$Port = 10088,
    [int]$ContextSize = 212992,
    [ValidateSet('f8_e4m3', 'q8_0', 'f16', 'q4_0')]
    [string]$KvType = 'f8_e4m3',
    [ValidateSet('low', 'medium', 'xhigh')]
    [string]$ReasoningEffort = 'medium',
    [int]$DraftMax = 3,
    [int]$ParallelSlots = 1,
    [int]$CacheRamMiB = 8192
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeDir) {
    if ($env:LOCAL_LLM_RUNTIME_DIR) {
        $RuntimeDir = $env:LOCAL_LLM_RUNTIME_DIR
    } elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'llama-server.exe')) {
        $RuntimeDir = $PSScriptRoot
    } else {
        $RuntimeDir = Join-Path $RepoRoot 'runtime\llama-cpp'
    }
}
if (-not $ModelsDir) {
    $ModelsDir = if ($env:LOCAL_LLM_MODELS_DIR) { $env:LOCAL_LLM_MODELS_DIR } else { Join-Path $RepoRoot 'models' }
}

$ServerExe = Join-Path $RuntimeDir 'llama-server.exe'
if (-not $ModelPath) {
    if ($env:LOCAL_LLM_MODEL_PATH) {
        $ModelPath = $env:LOCAL_LLM_MODEL_PATH
    } else {
        $Candidates = @(
            'Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf',
            'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
        )
        foreach ($Name in $Candidates) {
            $CandidatePath = Join-Path $ModelsDir $Name
            if (Test-Path -LiteralPath $CandidatePath) {
                $ModelPath = $CandidatePath
                break
            }
        }
        if (-not $ModelPath) {
            $FirstGguf = Get-ChildItem -LiteralPath $ModelsDir -Filter '*.gguf' -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notlike 'mmproj*' } |
                Select-Object -First 1
            if ($FirstGguf) { $ModelPath = $FirstGguf.FullName }
        }
    }
}

$ApiKey = if ($env:LLAMA_API_KEY) { $env:LLAMA_API_KEY } else { 'local-dev-key' }

if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "Missing llama-server.exe at '$ServerExe'. Run .\scripts\setup.ps1 or pass -RuntimeDir."
}
if (-not $ModelPath -or -not (Test-Path -LiteralPath $ModelPath)) {
    throw "No model .gguf found in '$ModelsDir'. Run .\scripts\setup.ps1 or pass -ModelPath."
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the existing server or pass -Port."
}

$ServerArgs = @(
    '-m', $ModelPath,
    '--alias', $Alias,
    '--api-key', $ApiKey,
    '--host', '127.0.0.1',
    '--port', "$Port",
    '-fa', 'on',
    '-ngl', 'all',
    '--fit', 'off',
    '-c', "$ContextSize",
    '-np', "$ParallelSlots",
    '-ctk', $KvType,
    '-ctv', $KvType,
    '-b', '2048',
    '-ub', '512',
    '-n', '65536',
    '--cache-ram', "$CacheRamMiB",
    '--ctx-checkpoints', '8',
    '--jinja',
    '--reasoning-format', 'deepseek',
    '--reasoning-effort', $ReasoningEffort,
    '--reasoning-preserve',
    '--temp', '0.6',
    '--top-k', '20',
    '--top-p', '0.95',
    '--min-p', '0.0',
    '--repeat-penalty', '1.0',
    '--presence-penalty', '0.0'
)

if ($ParallelSlots -ge 2) {
    $ServerArgs += '--no-cache-idle-slots'
} elseif ($DraftMax -gt 0) {
    # shortcut: MTP speculative decoding accelerates single-slot (-np 1) streams; skipped when ParallelSlots >= 2
    $ServerArgs += @(
        '-ngld', 'all',
        '-ctkd', 'q8_0',
        '-ctvd', 'q8_0',
        '--spec-type', 'draft-mtp',
        '--spec-draft-n-max', "$DraftMax"
    )
}

Write-Host "Starting $Alias on http://127.0.0.1:$Port/v1 (ctx=$ContextSize, slots=$ParallelSlots, kv=$KvType, effort=$ReasoningEffort)..." -ForegroundColor Cyan
& $ServerExe @ServerArgs
