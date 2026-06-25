# ============================================================
# 准备复赛镜像构建上下文：把"裁剪后的"模型权重 + checkpoint + 阈值
# 复制到仓库根目录的 models/ ckpt/ thresholds/，供 Dockerfile COPY。
# 源权重目录保持不动（只复制需要的文件，跳过 ~21GB 冗余格式）。
#
# 用法（仓库根目录）：
#   powershell -ExecutionPolicy Bypass -File scripts\stage_submission.ps1
# ============================================================
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$WhisperSrc = "D:\bisai\modelscope\whisper-large-v3"
$QwenSrc    = "D:\bisai\modelscope\Qwen3-0.6B"
$CkptSrc    = Join-Path $RepoRoot "outputs\lmf_vapfeat\checkpoints\ensemble_ep6.pt"
$ThrSrc     = Join-Path $RepoRoot "outputs\lmf_vapfeat\logs\best_thresholds.json"

# whisper-large-v3：只保留 fp16 单文件权重 + 配置/分词器（丢弃 flax/fp32/bin 等冗余）
$WhisperKeep = @(
  "model.safetensors",
  "config.json", "generation_config.json", "preprocessor_config.json",
  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
  "added_tokens.json", "special_tokens_map.json", "normalizer.json"
)

# Qwen3-0.6B：标准 HF 文件（已是单一 safetensors）
$QwenKeep = @(
  "model.safetensors",
  "config.json", "generation_config.json",
  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"
)

function Copy-Subset($srcDir, $dstDir, $keep) {
  if (-not (Test-Path $srcDir)) { throw "源目录不存在: $srcDir" }
  New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
  foreach ($f in $keep) {
    $s = Join-Path $srcDir $f
    if (Test-Path $s) {
      Copy-Item $s (Join-Path $dstDir $f) -Force
      Write-Host ("  [ok] {0}" -f $f)
    } else {
      Write-Host ("  [warn] 缺失(跳过): {0}" -f $f) -ForegroundColor Yellow
    }
  }
}

Write-Host "==> staging whisper-large-v3"
Copy-Subset $WhisperSrc (Join-Path $RepoRoot "models\whisper-large-v3") $WhisperKeep

Write-Host "==> staging Qwen3-0.6B"
Copy-Subset $QwenSrc (Join-Path $RepoRoot "models\Qwen3-0.6B") $QwenKeep

Write-Host "==> staging checkpoint + thresholds"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "ckpt") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "thresholds") | Out-Null
if (-not (Test-Path $CkptSrc)) { throw "checkpoint 不存在: $CkptSrc" }
if (-not (Test-Path $ThrSrc))  { throw "阈值文件不存在: $ThrSrc" }
Copy-Item $CkptSrc (Join-Path $RepoRoot "ckpt\ensemble_ep6.pt") -Force
Copy-Item $ThrSrc  (Join-Path $RepoRoot "thresholds\best_thresholds.json") -Force

Write-Host ""
Write-Host "==> staged. sizes:"
"models\whisper-large-v3", "models\Qwen3-0.6B", "ckpt", "thresholds" | ForEach-Object {
  $p = Join-Path $RepoRoot $_
  $mb = (Get-ChildItem $p -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
  Write-Host ("  {0,-28} {1,8:N1} MB" -f $_, $mb)
}
Write-Host ""
Write-Host "下一步: docker build -t finvcup-infer:v1 ."
