$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$RunName     = "lmf_dualvapbc_gated_5090"
$CkptSrcDir  = Join-Path $RepoRoot "outputs\$RunName\checkpoints"
$ManifestSrc = Join-Path $RepoRoot "outputs\$RunName\logs\ensemble_manifest.json"
$CpcWeight   = Join-Path $RepoRoot "models\60k_epoch4-d0f474de.pt"
$VapWeight   = Join-Path $RepoRoot "models\vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"
$BcWeight    = Join-Path $RepoRoot "models\bc_state_dict_ch_10hz_20000msec.pt"
$WhisperDir  = Join-Path $RepoRoot "models\whisper-large-v3"
$QwenDir     = Join-Path $RepoRoot "models\Qwen3-0.6B"

Write-Host "==> verifying backbones / weights in models\"
foreach ($p in @($WhisperDir, $QwenDir, $CpcWeight, $VapWeight, $BcWeight)) {
  if (Test-Path $p) { Write-Host "  [ok] $p" }
  else { Write-Host "  [ERR] missing $p" -ForegroundColor Red; exit 1 }
}
foreach ($f in @("model.safetensors","config.json")) {
  if (-not (Test-Path (Join-Path $WhisperDir $f))) { Write-Host "  [ERR] whisper missing $f" -ForegroundColor Red; exit 1 }
  if (-not (Test-Path (Join-Path $QwenDir $f)))    { Write-Host "  [ERR] qwen missing $f"    -ForegroundColor Red; exit 1 }
}

Write-Host "==> staging $RunName ensemble checkpoints + manifest -> ckpt\"
$CkptDst = Join-Path $RepoRoot "ckpt"
New-Item -ItemType Directory -Force -Path $CkptDst | Out-Null

$members = @()
if (Test-Path $ManifestSrc) {
  $manifest = Get-Content $ManifestSrc -Raw | ConvertFrom-Json
  foreach ($m in $manifest.members) {
    if ($m.name) { $members += [string]$m.name }
  }
}
if ($members.Count -eq 0) {
  $members = Get-ChildItem $CkptSrcDir -Filter "ensemble_ep*.pt" | Sort-Object Name | Select-Object -ExpandProperty Name
}
if ($members.Count -eq 0) {
  Write-Host "  [ERR] no ensemble_ep*.pt under $CkptSrcDir" -ForegroundColor Red
  exit 1
}

foreach ($name in $members) {
  $s = Join-Path $CkptSrcDir $name
  $d = Join-Path $CkptDst $name
  if (-not (Test-Path $s)) { Write-Host "  [ERR] missing checkpoint: $s" -ForegroundColor Red; exit 1 }
  Copy-Item $s $d -Force
  Write-Host "  [ok] $name"
}
if (-not (Test-Path $ManifestSrc)) { Write-Host "  [ERR] missing manifest: $ManifestSrc" -ForegroundColor Red; exit 1 }
Copy-Item $ManifestSrc (Join-Path $CkptDst "ensemble_manifest.json") -Force
Write-Host "  [ok] ensemble_manifest.json"

Write-Host ""
Write-Host "Next:"
Write-Host "  docker build -f Dockerfile.ensemble_vap -t finvcup-infer:codex-dualch-vapwin-bc-gated ."
