# 复赛镜像提交说明（ensemble 方案）

本目录已按官方《复赛镜像提交说明》适配好集成推理镜像。**镜像必须在训练服务器上构建**
（模型权重、ensemble 成员 checkpoint、manifest 都在服务器上，不在 git 里）。

## 文件清单

| 文件 | 作用 |
|------|------|
| `run.sh` | 镜像入口，容器内 `bash run.sh`，读 `/xydata` → 写 `/app/submit/submit.csv` |
| `Dockerfile.baseline` | 构建推理镜像（base = 官方 baseline `finvcup/team35:finv_baseline`，torch2.5.1/py3.11/transformers4.57.6；只补 sklearn 等少量包） |
| `configs/docker_infer_ensemble.yaml` | 容器内 `/app` 路径配置（结构字段复制自 tuned 训练配置） |
| `scripts/build_submit_image.sh` | 暂存产物 → 构建镜像（一条命令；需在能正常 docker build 的机器上跑） |
| `test_data/` | 10 条样例，仅用于本地自测（已被 `.dockerignore` 排除，不进镜像） |

## 一键构建（在服务器仓库根目录）

```bash
bash scripts/build_submit_image.sh
```

默认源路径（可用环境变量覆盖，见脚本头部注释）：
- 训练产物：`outputs/lmf_ensemble_4xL20_tuned/{checkpoints, logs/ensemble_manifest.json}`
- 模型权重：`/mnt/workspace/dorihue/modelscope/{whisper-large-v3, Qwen3-0.6B}`

换其他 run，例如：
```bash
RUN_DIR=outputs/lmf_ensemble_4xL20 IMAGE_TAG=finvcup-infer:v2 bash scripts/build_submit_image.sh
```

## 本地自测（务必先跑通再提交）

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/test_data:/xydata:ro \
  -v $(pwd)/_submit_out:/app/submit \
  finvcup-infer:latest bash -lc 'bash run.sh && head /app/submit/submit.csv'
```
期望输出表头：`segment_id,c,na,i,bc,t`，10 行结果。

## 推送到比赛 Registry（复赛阶段一）

```bash
docker login --username=<短信里的用户名> finvcup-registry.cn-shanghai.cr.aliyuncs.com
docker tag finvcup-infer:latest finvcup-registry.cn-shanghai.cr.aliyuncs.com/finvcup/<仓库名>:<版本号>
docker push finvcup-registry.cn-shanghai.cr.aliyuncs.com/finvcup/<仓库名>:<版本号>
```
然后在后台「提交作品(复赛)」填镜像地址提交。每天最多 2 次。

## ⚠️ 提交前必查（每条都会直接影响能否跑分）

1. **结构字段一致**：`configs/docker_infer_ensemble.yaml` 里 `audio_encoder / text_encoder /
   context_encoder / fusion / labels / chunk` 必须和「训练出该 checkpoint 的配置」完全相同。
   本文件复制自 `whisper_qwen0_6b_lmf_ensemble_4xL20_tuned.yaml`；若提交别的 run 的 checkpoint，
   请同步改成那个 run 的结构字段，否则 `strict=False` 会静默错配 → 跑分变乱码。
2. **transformers 版本一致**：构建脚本会自动探测当前 python 环境的版本注入。请确保在
   **训练用的同一个环境**里执行构建脚本（`unfreeze_layers:2` 推理会走 `_forward_encoder_split`，
   依赖该版本 WhisperEncoder 内部实现）。
3. **镜像大小** ≤ 20GB（提交上限）：`docker images finvcup-infer:latest` 确认。
4. **参数量** ≤ 8B：单成员 ≈ whisper encoder(~0.6B) + Qwen3-0.6B + 小头 ≈ 1.2B，
   推理时一次只载一个成员，合规。
5. **推理超时** ≤ 60 分钟：测试集大时如超时，调小 `eval_batch_size` 或 `TOPK`（用更少成员）。
