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

Write-Host "==> validating first checkpoint architecture"
$firstName = [string]$members[0].name
$firstPath = Join-Path $CkptSrcDir $firstName
$validateCode = @"
import sys, torch
path = sys.argv[1]
ck = torch.load(path, map_location='cpu')
cfg = ck.get('config', {})
vf = cfg.get('vap_feat', {})
audio = cfg.get('audio_encoder', {})
if int(vf.get('feat_dim', -1)) != 21:
    raise SystemExit(f'[ERR] {path} checkpoint vap_feat.feat_dim={vf.get("feat_dim")} != 21')
if not bool((audio.get('stereo_branch') or {}).get('enabled', False)):
    raise SystemExit(f'[ERR] {path} checkpoint stereo_branch is not enabled')
sd = ck.get('model', {})
w = sd.get('vap_feat_encoder.conv.0.weight')
if w is None or tuple(w.shape)[1] != 21:
    raise SystemExit(f'[ERR] {path} missing 21d vap_feat_encoder.conv.0.weight, got {None if w is None else tuple(w.shape)}')
print(f'[ok] checkpoint config/output matches dualvapbc: {path}')
"@
python -c $validateCode $firstPath

Write-Host "==> staging ensemble checkpoints + manifest -> ckpt_submit\"
$CkptDst = Join-Path $RepoRoot "ckpt_submit"
if (Test-Path $CkptDst) {
  Get-ChildItem $CkptDst -File | Remove-Item -Force
}
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
