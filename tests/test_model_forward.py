"""Shape + gradient smoke tests for collate, the network and a training step."""

import numpy as np
import torch

from src.data.dataset import Bag, collate_bags, make_group_id
from src.losses.group_dro import GroupDROLoss
from src.models.encoders import pool_segments_to_bags
from src.models.transval_net import TransValNet
from src.train.trainer import Trainer, TrainerConfig, gather_seg_labels


def _toy_segment(with_visual: bool = True):
    return {
        "audio": np.random.randn(7, 1024).astype("float32"),
        "acoustic": np.random.randn(5, 79).astype("float32"),
        "text": np.random.randn(768).astype("float32"),
        "visual": np.random.randn(4, 50).astype("float32") if with_visual else None,
        "qtype": 3,
        "corpus_id": 0,
        "gender_id": 1,
    }


def _toy_bags(n_bags: int = 4):
    bags = []
    for b in range(n_bags):
        segs = [_toy_segment(with_visual=(b % 2 == 0)) for _ in range(3)]
        bags.append(
            Bag(
                participant_id=f"p{b}",
                label=b % 2,
                group_id=make_group_id(0, b % 2),
                segments=segs,
            )
        )
    return bags


def test_collate_shapes():
    batch = collate_bags(_toy_bags(4))
    n_seg = batch["seg2bag"].numel()
    assert batch["audio"].shape[0] == n_seg
    assert batch["text"].shape == (n_seg, 768)
    assert batch["modality_mask"].shape == (n_seg, 4)
    assert batch["bag_labels"].shape == (4,)
    assert batch["seg2bag"].max().item() == 3


def test_forward_shapes():
    batch = collate_bags(_toy_bags(4))
    net = TransValNet(d=32, n_corpus=3, use_adv=True)
    out = net(batch)
    assert out["logit_bag"].shape == (4,)
    assert out["logit_seg"].shape == (batch["seg2bag"].numel(),)
    assert out["gate_bag"].shape == (4,)
    assert out["corpus_logit"].shape[0] == batch["seg2bag"].numel()


def test_pool_segments_to_bags_partition():
    z = torch.randn(6, 8)
    seg2bag = torch.tensor([0, 0, 1, 1, 2, 2])
    attn = torch.nn.Linear(8, 1)
    z_bag = pool_segments_to_bags(z, seg2bag, 3, attn)
    assert z_bag.shape == (3, 8)
    assert torch.isfinite(z_bag).all()


def test_group_dro_step():
    loss = GroupDROLoss(4)
    per_sample = torch.rand(8)
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    val = loss(per_sample, groups)
    assert val.ndim == 0
    assert abs(float(loss.q.sum()) - 1.0) < 1e-5


def test_training_step_runs_and_backprops():
    bags = _toy_bags(6)
    batch = collate_bags(bags)
    net = TransValNet(d=16, n_corpus=3, use_adv=True)
    cfg = TrainerConfig(epochs=1, use_group_dro=True, use_irm=True, device="cpu")
    trainer = Trainer(net, cfg, n_groups=6)
    out = net(batch)
    loss = trainer._compute_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_gather_seg_labels_broadcast():
    batch = collate_bags(_toy_bags(3))
    seg_labels = gather_seg_labels(batch)
    assert seg_labels.shape[0] == batch["seg2bag"].numel()
    # Each segment label equals its bag's label.
    for i, b in enumerate(batch["seg2bag"].tolist()):
        assert seg_labels[i].item() == batch["bag_labels"][b].item()
