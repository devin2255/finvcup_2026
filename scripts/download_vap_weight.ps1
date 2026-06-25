# ============================================================
# 下载 vap_mc_ch_kyoto 运行时权重到本地 models/
# 来源： maai-kyoto/vap_mc_ch_kyoto 的
#       vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt
# （与 configs/submit_ensemble_vap.yaml 中的 vap_feat.frame_rate=10,
#   context_sec=20, lang=ch_kyoto, mode=vap_mc 对应。）
#
# 用法（仓库根目录）：
#   powershell -File scripts\download_vap_weight.ps1
# 默认走 hf-mirror.com，国内可访问；如果失败会自动尝试 huggingface.co。
# ============================================================

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DestDir  = Join-Path $RepoRoot "models"
$DestFile = Join-Path $DestDir "vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

if (Test-Path $DestFile) {
  $sizeMB = (Get-Item $DestFile).Length / 1MB
  Write-Host ("[skip] already present: {0} ({1:N1} MB)" -f $DestFile, $sizeMB)
  exit 0
}

$relPath = "maai-kyoto/vap_mc_ch_kyoto/resolve/main/vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt"
$urls = @(
  "https://hf-mirror.com/$relPath",
  "https://huggingface.co/$relPath"
)

$ok = $false
foreach ($u in $urls) {
  Write-Host "==> trying $u"
  try {
    Invoke-WebRequest -Uri $u -OutFile $DestFile -UseBasicParsing -TimeoutSec 600
    $ok = $true
    break
  } catch {
    Write-Host ("  [fail] {0}" -f $_.Exception.Message) -ForegroundColor Yellow
    if (Test-Path $DestFile) { Remove-Item $DestFile -Force }
  }
}

if (-not $ok) {
  Write-Host "ERROR: 所有镜像都失败。请手动下载 vap_mc_state_dict_ch_kyoto_10hz_20000msec.pt 到 models\" -ForegroundColor Red
  exit 1
}

$sizeMB = (Get-Item $DestFile).Length / 1MB
Write-Host ""
Write-Host ("[ok] downloaded -> {0} ({1:N1} MB)" -f $DestFile, $sizeMB) -ForegroundColor Green
