param(
    [ValidateSet('uncensored', 'clean')]
    [string]$Profile = 'uncensored',

    [string]$RuntimeDir = '',
    [string]$ModelsDir = '',
    [string]$LlamaTag = 'b12359'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeDir) {
    $RuntimeDir = if ($env:LOCAL_LLM_RUNTIME_DIR) { $env:LOCAL_LLM_RUNTIME_DIR } else { Join-Path $RepoRoot 'runtime\llama-cpp' }
}
if (-not $ModelsDir) {
    $ModelsDir = if ($env:LOCAL_LLM_MODELS_DIR) { $env:LOCAL_LLM_MODELS_DIR } else { Join-Path $RepoRoot 'models' }
}

$ModelSpec = if ($Profile -eq 'uncensored') {
    @{
        Repo   = 'HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF'
        File   = 'Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf'
        Sha256 = '582b80812df428af61cd77891c703b4f0ac388fc22759a159a6c408886ba7a2b'
    }
} else {
    @{
        Repo   = 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF'
        File   = 'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
        Sha256 = '58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f'
    }
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

$ServerExe = Join-Path $RuntimeDir 'llama-server.exe'
if (-not (Test-Path -LiteralPath $ServerExe)) {
    Write-Host "[1/3] Downloading llama.cpp $LlamaTag (win-cuda-12.4-x64) to $RuntimeDir ..." -ForegroundColor Cyan
    $Assets = @(
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-$LlamaTag-bin-win-cuda-12.4-x64.zip"
    )
    $GhCmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($GhCmd) {
        & $GhCmd.Source release download $LlamaTag --repo ggml-org/llama.cpp `
            --pattern $Assets[0] --pattern $Assets[1] --dir $RuntimeDir --clobber
    } else {
        foreach ($Asset in $Assets) {
            $Url = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/$Asset"
            $OutZip = Join-Path $RuntimeDir $Asset
            Invoke-WebRequest -Uri $Url -OutFile $OutZip -UseBasicParsing
        }
    }

    foreach ($Asset in $Assets) {
        Expand-Archive -LiteralPath (Join-Path $RuntimeDir $Asset) -DestinationPath $RuntimeDir -Force
    }
} else {
    Write-Host "[1/3] Found existing llama-server.exe at $ServerExe" -ForegroundColor Green
}

Write-Host "[2/3] Verifying CUDA device visibility ..." -ForegroundColor Cyan
& $ServerExe --list-devices

$TargetModel = Join-Path $ModelsDir $ModelSpec.File
if (-not (Test-Path -LiteralPath $TargetModel)) {
    Write-Host "[3/3] Downloading $($ModelSpec.File) from $($ModelSpec.Repo) ..." -ForegroundColor Cyan
    $HfCmd = Get-Command hf -ErrorAction SilentlyContinue
    $UvxCmd = Get-Command uvx -ErrorAction SilentlyContinue

    if ($HfCmd) {
        & $HfCmd.Source download $ModelSpec.Repo $ModelSpec.File --local-dir $ModelsDir
    } elseif ($UvxCmd) {
        & $UvxCmd.Source --from huggingface_hub hf download $ModelSpec.Repo $ModelSpec.File --local-dir $ModelsDir
    } else {
        $ModelUrl = "https://huggingface.co/$($ModelSpec.Repo)/resolve/main/$($ModelSpec.File)"
        Invoke-WebRequest -Uri $ModelUrl -OutFile $TargetModel -UseBasicParsing
    }
} else {
    Write-Host "[3/3] Found existing model file at $TargetModel" -ForegroundColor Green
}

Write-Host "Verifying SHA-256 checksum ..." -ForegroundColor Cyan
$ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $TargetModel).Hash.ToLower()
if ($ActualHash -ne $ModelSpec.Sha256) {
    throw "SHA-256 mismatch for $TargetModel. Expected $($ModelSpec.Sha256), got $ActualHash"
}
Write-Host "SHA-256 OK: $ActualHash" -ForegroundColor Green
Write-Host "`nSetup complete. Next: run .\scripts\start-server.ps1" -ForegroundColor Green
