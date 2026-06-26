# ============================================================
# Stage build context for dualch v2 image (ensemble + real single-frame VAP,
# dual-channel audio variant).
#
# Difference vs stage_submission_ensemble.ps1: checkpoints/manifest come from
# outputs\lmf_dualch (the stereo-branch model). Backbones (whisper/Qwen), CPC
# and VAP weights already live under models\; this script only verifies them
# and refreshes ckpt\.
#
# IMPORTANT: build MUST be on branch feat/dualch-audio (src\ has
# StereoActivityEncoder + DualChannelAudioEncoder), and
# configs\submit_ensemble_vap.yaml must have audio_encoder.stereo_branch.enabled
# = true (matching training) or the stereo params will fail to load.
# dualch uses single-frame VAP (NOT window); precompute stays baseline.
#
# Usage (repo root, after: git checkout feat/dualch-audio):
#   powershell -File scripts\stage_submission_dualch.ps1
# ============================================================
$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$CkptSrcDir  = Join-Path $RepoRoot "outputs\lmf_dualch\checkpoints"
$ManifestSrc = Join-Path $RepoRoot "outputs\lmf_dualch\logs\ensemble_manifest.json"
$CpcWeight   = Join-Path $RepoRoot "models\60k_epoch4-d0f474de.pt"
$VapWeight   = Join-Path $RepoRoot "models\vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"
$WhisperDir  = Join-Path $RepoRoot "models\whisper-large-v3"
$QwenDir     = Join-Path $RepoRoot "models\Qwen3-0.6B"

Write-Host "==> verifying backbones / weights in models\"
foreach ($p in @($WhisperDir, $QwenDir, $CpcWeight, $VapWeight)) {
  if (Test-Path $p) { Write-Host "  [ok] $p" }
  else { Write-Host "  [ERR] missing $p" -ForegroundColor Red; exit 1 }
}
foreach ($f in @("model.safetensors","config.json")) {
  if (-not (Test-Path (Join-Path $WhisperDir $f))) { Write-Host "  [ERR] whisper missing $f" -ForegroundColor Red; exit 1 }
  if (-not (Test-Path (Join-Path $QwenDir $f)))    { Write-Host "  [ERR] qwen missing $f"    -ForegroundColor Red; exit 1 }
}

Write-Host "==> staging dualch ensemble checkpoints + manifest -> ckpt\"
$CkptDst = Join-Path $RepoRoot "ckpt"
New-Item -ItemType Directory -Force -Path $CkptDst | Out-Null
foreach ($name in @("ensemble_ep3.pt","ensemble_ep4.pt","ensemble_ep5.pt","ensemble_ep6.pt","ensemble_ep7.pt")) {
  $s = Join-Path $CkptSrcDir $name
  $d = Join-Path $CkptDst $name
  if (-not (Test-Path $s)) { Write-Host "  [ERR] missing checkpoint: $s" -ForegroundColor Red; exit 1 }
  Copy-Item $s $d -Force; Write-Host "  [ok] $name"   # overwrite old members (vapfeat/vapwin)
}
if (-not (Test-Path $ManifestSrc)) { Write-Host "  [ERR] missing manifest: $ManifestSrc" -ForegroundColor Red; exit 1 }
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
Write-Host "Confirm: git branch must be feat/dualch-audio; submit_ensemble_vap.yaml has stereo_branch enabled."
Write-Host "Next: docker build -f Dockerfile.ensemble_vap -t finvcup-infer:dualch ."
