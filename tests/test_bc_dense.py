import numpy as np
import torch

from src.bc_dense import bc_dense_target


# --- bc_dense_target: 未来窗口逐 chunk BC 目标 ---

def test_dense_target_marks_bc_chunks():
    labels = np.array([0, 0, 2, 1, 2, 4, 0, 0], dtype=np.int64)
    out = bc_dense_target(labels, end_idx=1, target_chunks=5, bc_id=2)
    assert out.dtype == np.float32 and out.shape == (5,)
    np.testing.assert_array_equal(out, [0, 1, 0, 1, 0])  # labels[1:6] == [0,2,1,2,4]


def test_dense_target_all_zero_when_no_bc():
    labels = np.zeros(40, dtype=np.int64)
    out = bc_dense_target(labels, end_idx=10, target_chunks=25, bc_id=2)
    assert out.shape == (25,) and not out.any()


def test_dense_target_pads_right_when_short():
    labels = np.array([2, 2, 2], dtype=np.int64)
    out = bc_dense_target(labels, end_idx=1, target_chunks=5, bc_id=2)
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, [1, 1, 0, 0, 0])  # 只有 2 个真实 chunk，右侧补 0


def test_dense_target_window_any_equals_window_label():
    # 密集目标的 any() 必须与窗口级 BC 标签(build_train_samples_multitask 的口径)一致
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 5, size=200).astype(np.int64)
    for end_idx in (10, 50, 175):
        dense = bc_dense_target(labels, end_idx, 25, bc_id=2)
        window_label = int((labels[end_idx: end_idx + 25] == 2).any())
        assert int(dense.any()) == window_label


# --- bc_dense 头：与模型 head 相同的纯 torch 结构，验证形状与 max 聚合诊断 ---

def test_bc_dense_head_shape_and_window_aggregation():
    hidden, chunks, B = 320, 25, 4
    head = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, chunks),
    )
    fused = torch.randn(B, hidden)
    logits = head(fused)
    assert logits.shape == (B, chunks)
    win_prob = torch.sigmoid(logits).amax(dim=1)
    assert win_prob.shape == (B,)
    assert bool(((win_prob >= 0) & (win_prob <= 1)).all())


def test_bc_dense_loss_pos_weight_upweights_positives():
    # pos_weight>1 时，正例误判的损失应大于对称的负例误判
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))
    logits = torch.tensor([[-3.0]])
    miss_pos = crit(logits, torch.tensor([[1.0]]))   # 正例被预测为负
    miss_neg = crit(-logits, torch.tensor([[0.0]]))  # 负例被预测为正（对称）
    assert miss_pos > miss_neg
