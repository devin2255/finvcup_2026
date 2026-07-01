from pathlib import Path


def test_dualvapbc_docker_and_stage_use_same_ckpt_dir():
    dockerfile = Path("Dockerfile.ensemble_vap").read_text(encoding="utf-8")
    stage = Path("scripts/stage_submission_dualvapbc.ps1").read_text(encoding="utf-8")

    assert "COPY ckpt_submit/ /app/ckpt/" in dockerfile
    assert '$CkptDst = Join-Path $RepoRoot "ckpt_submit"' in stage
    assert "vap_feat.feat_dim" in stage
    assert "vap_feat_encoder.conv.0.weight" in stage


def test_dualvapbc_run_defaults_are_submission_safe():
    run = Path("run.ensemble_vap.sh").read_text(encoding="utf-8")

    assert 'VAP_WORKERS:-4' in run
    assert 'ENSEMBLE_TOPK=${ENSEMBLE_TOPK:-3}' in run
    assert '--bc_enabled' in run
    assert '--topk "${ENSEMBLE_TOPK}"' in run
