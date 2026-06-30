# Stage dualch + vapwin + BC-tail ensemble assets for Docker submission.
$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$OutName     = "lmf_dualvapbc_5090"
$CkptSrcDir  = Join-Path $RepoRoot "outputs\$OutName\checkpoints"
$ManifestSrc = Join-Path $RepoRoot "outputs\$OutName\logs\ensemble_manifest.json"
$CpcWeight   = Join-Path $RepoRoot "models\60k_epoch4-d0f474de.pt"
$VapWeight   = Join-Path $RepoRoot "models\vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"
$BcWeight    = Join-Path $RepoRoot "models\vap-bc_state_dict_ch_10hz_20000msec.pt"
$WhisperDir  = Join-Path $RepoRoot "models\whisper-large-v3"
$QwenDir     = Join-Path $RepoRoot "models\Qwen3-0.6B"

Write-Host "==> verifying backbones / MaAI weights"
foreach ($p in @($WhisperDir, $QwenDir, $CpcWeight, $VapWeight, $BcWeight)) {
  if (Test-Path $p) { Write-Host "  [ok] $p" }
  else { Write-Host "  [ERR] missing $p" -ForegroundColor Red; exit 1 }
}

if (-not (Test-Path $ManifestSrc)) {
  Write-Host "  [ERR] missing manifest: $ManifestSrc" -ForegroundColor Red
  exit 1
}

$manifest = Get-Content $ManifestSrc -Raw | ConvertFrom-Json
$members = @($manifest.members)
if ($members.Count -eq 0) {
  Write-Host "  [ERR] manifest has no members" -ForegroundColor Red
  exit 1
}

Write-Host "==> staging ensemble checkpoints + manifest -> ckpt\"
$CkptDst = Join-Path $RepoRoot "ckpt"
New-Item -ItemType Directory -Force -Path $CkptDst | Out-Null
foreach ($m in $members) {
  $name = $m.name
  $s = Join-Path $CkptSrcDir $name
  $d = Join-Path $CkptDst $name
  if (-not (Test-Path $s)) {
    Write-Host "  [ERR] missing checkpoint: $s" -ForegroundColor Red
    exit 1
  }
  Copy-Item $s $d -Force
  Write-Host ("  [ok] {0} epoch={1} metric={2}" -f $name, $m.epoch, $m.metric)
}
Copy-Item $ManifestSrc (Join-Path $CkptDst "ensemble_manifest.json") -Force
Write-Host "  [ok] ensemble_manifest.json"

Write-Host ""
Write-Host "Next:"
Write-Host "  docker build -f Dockerfile.ensemble_vap -t finvcup-infer:dualvapbc ."
