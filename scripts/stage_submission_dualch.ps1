# ============================================================
# 准备 dualch v2 镜像构建上下文（ensemble + 真单帧 VAP，双声道音频变体）。
#
# 与 stage_submission_ensemble.ps1 的区别：checkpoint/manifest 来自
# outputs\lmf_dualch（双声道 stereo 分支模型）。骨干权重(whisper/Qwen)、CPC、
# VAP 权重已在 models\ 下，本脚本只校验存在性 + 刷新 ckpt\。
#
# 重要：构建必须在 feat/dualch-audio 分支（src\ 含 StereoActivityEncoder +
# DualChannelAudioEncoder），且 configs\submit_ensemble_vap.yaml 已开
# audio_encoder.stereo_branch.enabled=true（与训练一致），否则 stereo 参数加载不上。
# dualch 用单帧 VAP（不是 window），precompute 走基线逻辑。
#
# 用法（仓库根目录，且已 git checkout feat/dualch-audio）：
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
  Copy-Item $s $d -Force; Write-Host "  [ok] $name"   # 覆盖旧成员（vapfeat/vapwin）
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
Write-Host "确认：git branch 必须是 feat/dualch-audio；submit_ensemble_vap.yaml 已开 stereo_branch。"
Write-Host "下一步: docker build -f Dockerfile.ensemble_vap -t finvcup-infer:dualch ."
