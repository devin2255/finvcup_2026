import torch
import yaml

from src.bc_residual import BcResidualHead, apply_label_residual


def test_bc_residual_head_outputs_batch_delta():
    head = BcResidualHead(feat_dim=3, hidden=8, dropout=0.0)
    out = head(torch.randn(4, 20, 3))
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_apply_label_residual_only_changes_target_column():
    logits = torch.randn(3, 5)
    delta = torch.tensor([[0.1], [0.2], [0.3]])
    out = apply_label_residual(logits, delta, target_index=3)

    assert torch.allclose(out[:, :3], logits[:, :3])
    assert torch.allclose(out[:, 4:], logits[:, 4:])
    assert torch.allclose(out[:, 3], logits[:, 3] + delta.squeeze(-1))


def test_gated_bc_configs_keep_bc_out_of_main_vap_encoder():
    for path in [
        "configs/whisper_qwen0_6b_lmf_dualvapbc_gated_5090.yaml",
        "configs/submit_ensemble_vap.yaml",
    ]:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vf = cfg["vap_feat"]
        bc = vf["bc_residual"]
        assert vf["feat_dim"] == 21
        assert bc["enabled"] is True
        assert bc["start"] == 18
        assert bc["feat_dim"] == 3
        assert bc["base_vap_dim"] == 18
        assert cfg["audio_encoder"]["stereo_branch"]["enabled"] is True
        assert cfg["train"]["pos_weight_mode"] == "capped_per_label"
        assert "pos_weight_cap_per_label" not in cfg["train"]
