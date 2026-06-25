# ============================================================
# 准备 v2 镜像构建上下文（ensemble + 真 VAP）：
#   models\:
#     whisper-large-v3\        裁剪后的 fp16 safetensors + 配置（同 v1）
#     Qwen3-0.6B\              （同 v1）
#     60k_epoch4-d0f474de.pt   CPC 权重（用户已下载到 models\）
#     vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt   VAP 权重（download_vap_weight.ps1 下载）
#   ckpt\:
#     ensemble_ep3.pt / ep4.pt / ep5.pt / ep6.pt / ep7.pt
#     ensemble_manifest.json
#   .dockerignore-friendly：MaAI\ 已在仓库内，由 Dockerfile 直接 COPY。
#
# 用法（仓库根目录）：
#   powershell -File scripts\stage_submission_ensemble.ps1
# 已存在的文件不会重复复制。
# ============================================================
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$WhisperSrc = "D:\bisai\modelscope\whisper-large-v3"
$QwenSrc    = "D:\bisai\modelscope\Qwen3-0.6B"
$CkptSrcDir = Join-Path $RepoRoot "outputs\lmf_vapfeat\checkpoints"
$ManifestSrc = Join-Path $RepoRoot "outputs\lmf_vapfeat\logs\ensemble_manifest.json"
$CpcWeight  = Join-Path $RepoRoot "models\60k_epoch4-d0f474de.pt"
$VapWeight  = Join-Path $RepoRoot "models\vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"

$WhisperKeep = @(
  "model.safetensors",
  "config.json", "generation_config.json", "preprocessor_config.json",
  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
  "added_tokens.json", "special_tokens_map.json", "normalizer.json"
)
$QwenKeep = @(
  "model.safetensors",
  "config.json", "generation_config.json",
  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"
)

function Copy-Subset($srcDir, $dstDir, $keep) {
  if (-not (Test-Path $srcDir)) { throw "src not found: $srcDir" }
  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
  foreach ($f in $keep) {
    $s = Join-Path $srcDir $f
    $d = Join-Path $dstDir $f
    if (Test-Path $s) {
      if (Test-Path $d) { Write-Host "  [skip-exists] $f" } else { Copy-Item $s $d -Force; Write-Host "  [ok] $f" }
    } else { Write-Host "  [warn missing] $f" -ForegroundColor Yellow }
  }
}

Write-Host "==> whisper-large-v3"
Copy-Subset $WhisperSrc (Join-Path $RepoRoot "models\whisper-large-v3") $WhisperKeep

Write-Host "==> Qwen3-0.6B"
Copy-Subset $QwenSrc (Join-Path $RepoRoot "models\Qwen3-0.6B") $QwenKeep

Write-Host "==> CPC weight"
if (Test-Path $CpcWeight) {
  Write-Host ("  [ok] {0} ({1:N1} MB)" -f $CpcWeight, ((Get-Item $CpcWeight).Length/1MB))
} else {
  Write-Host "  [ERR] missing $CpcWeight" -ForegroundColor Red; exit 1
}

Write-Host "==> VAP weight"
if (Test-Path $VapWeight) {
  Write-Host ("  [ok] {0} ({1:N1} MB)" -f $VapWeight, ((Get-Item $VapWeight).Length/1MB))
} else {
  Write-Host "  [ERR] missing $VapWeight — please run scripts\download_vap_weight.ps1 first" -ForegroundColor Red
  exit 1
}

Write-Host "==> ensemble checkpoints + manifest"
$CkptDst = Join-Path $RepoRoot "ckpt"
New-Item -ItemType Directory -Force -Path $CkptDst | Out-Null
foreach ($name in @("ensemble_ep3.pt","ensemble_ep4.pt","ensemble_ep5.pt","ensemble_ep6.pt","ensemble_ep7.pt")) {
  $s = Join-Path $CkptSrcDir $name
  $d = Join-Path $CkptDst $name
  if (-not (Test-Path $s)) { Write-Host "  [ERR] missing checkpoint: $s" -ForegroundColor Red; exit 1 }
  if (Test-Path $d) { Write-Host "  [skip-exists] $name" } else { Copy-Item $s $d -Force; Write-Host "  [ok] $name" }
}
Copy-Item $ManifestSrc (Join-Path $CkptDst "ensemble_manifest.json") -Force
Write-Host "  [ok] ensemble_manifest.json"

Write-Host ""
Write-Host "==> sizes:"
"models", "models\whisper-large-v3", "models\Qwen3-0.6B", "ckpt", "MaAI" | ForEach-Object {
  $p = Join-Path $RepoRoot $_
  if (Test-Path $p) {
    $mb = (Get-ChildItem $p -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("  {0,-28} {1,9:N1} MB" -f $_, $mb)
  }
}
Write-Host ""
Write-Host "下一步: docker build -f Dockerfile.ensemble_vap -t finvcup-infer:v2-ens ."
