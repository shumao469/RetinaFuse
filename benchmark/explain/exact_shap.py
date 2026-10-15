"""Exact finite-background, interventional Shapley values (no SHAP dependency)."""
import math
import numpy as np

FEATURES = ['total_events', 'positive_events', 'negative_events', 'positive_fraction',
            'mean_interval', 'interval_sd', 'maximum_interval', 'mean_x', 'sd_x',
            'mean_y', 'sd_y', 'polarity_switches']
LABELS = ['Event count', 'Positive events', 'Negative events', 'Positive fraction',
          'Mean event interval', 'Interval SD', 'Maximum interval', 'Mean x',
          'SD x', 'Mean y', 'SD y', 'Polarity switches']
GROUPS = {'Abundance / polarity': [0, 1, 2, 3], 'Timestamp intervals': [4, 5, 6],
          'Spatial distribution': [7, 8, 9, 10], 'Polarity switching': [11]}


def shapley(values, n_features):
    """Coalition index bit j indicates that feature j belongs to that subset."""
    v = np.asarray(values, dtype=np.float64)
    if not 1 <= n_features <= 20 or v.shape != (1 << n_features,):
        raise ValueError('Expected one finite value for each feature coalition')
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite coalition values')
    masks = np.arange(len(v), dtype=np.int64)
    result = []
    for j in range(n_features):
        ix = masks[(masks & (1 << j)) == 0]
        weights = [1 / (n_features * math.comb(n_features - 1, int(i).bit_count())) for i in ix]
        result.append(np.dot(weights, v[ix | (1 << j)] - v[ix]))
    return np.asarray(result)


def group_masks(groups, n_features):
    """Map the grouped coalition game into the individual-feature game."""
    indices = [j for group in groups.values() for j in group]
    if sorted(indices) != list(range(n_features)):
        raise ValueError('Groups must partition the features without overlap')
    return np.array([sum(1 << j for k, group in enumerate(groups.values())
                         if mask & (1 << k) for j in group)
                     for mask in range(1 << len(groups))])


def select_donors(records, fold, healthy_count=4, mcao_count=12, seed=20261015):
    """Training identifiers only; exclude all held-out aliases, then deduplicate."""
    test = records[records.outer_fold.eq(fold)]
    held_families = set(test.source_family)
    train = records[(~records.outer_fold.eq(fold)) &
                    (~records.source_family.isin(held_families))]
    train = train.drop_duplicates('animal_id').drop_duplicates('source_family')
    rng = np.random.default_rng(seed + int(fold))
    donors = []
    for label, count in [(0, healthy_count), (1, mcao_count)]:
        ids = train[train.label.eq(label)].animal_id.sort_values().to_numpy()
        if count < 1 or len(ids) < count:
            raise ValueError(f'Insufficient unique training donors in fold {fold}, class {label}')
        donors.extend(rng.choice(ids, size=count, replace=False).tolist())
    return train.set_index('animal_id').loc[donors].reset_index()


def coalition_values(models, frame_embeddings, target, background, batch_size=256,
                     masks=None):
    """Average checkpoint margins, sigmoid, then average finite donor probabilities.

    The observed event-volume embedding stays fixed. The statistics encoder,
    gate and head are recomputed for every hybrid input. Unlike summing branch
    outputs, this includes the effect of descriptors on gate allocation.
    """
    import torch
    if len(models) != len(frame_embeddings) or not models or batch_size < 1:
        raise ValueError('Models/embeddings must be nonempty, matched; batch size positive')
    if background.ndim != 2 or target.shape != background.shape[1:] or len(background) < 1:
        raise ValueError('Expected target [features] and background [donors, features]')
    device = target.device
    indices = torch.arange(1 << target.numel(), device=device) if masks is None else torch.as_tensor(masks, device=device)
    bits = ((indices[:, None] >> torch.arange(target.numel(), device=device)) & 1).bool()
    result = []
    with torch.no_grad():
        for start in range(0, len(bits), batch_size):
            bit = bits[start:start + batch_size]
            x = torch.where(bit[:, None, :], target[None, None, :], background[None, :, :]).reshape(-1, target.numel())
            margins = []
            for model, hf in zip(models, frame_embeddings):
                hs = model.branch_b(x)
                ff = hf.reshape(1, -1).expand(len(x), -1)
                alpha = torch.softmax(model.fusion.gate(torch.cat([ff, hs], -1)), -1)
                logits = model.head(alpha[:, 0, None] * ff + alpha[:, 1, None] * hs)
                margins.append(logits[:, 1] - logits[:, 0])
            probabilities = torch.sigmoid(torch.stack(margins).mean(0))
            result.append(probabilities.reshape(len(bit), len(background)).mean(1).cpu().numpy())
    return np.concatenate(result).astype(np.float64)
