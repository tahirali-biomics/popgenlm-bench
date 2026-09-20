#!/usr/bin/env python3
import importlib.util
from pathlib import Path


def score(
    model,
    tokenizer,
    contexts,
    variants,
    device,
    batch_size=1,
    target_index=255,
    upstream_zero_shot_score=None,
):
    if upstream_zero_shot_score is None:
        raise ValueError("upstream zero_shot_score.py path is required")
    spec = importlib.util.spec_from_file_location(
        "plantcad_authors", Path(upstream_zero_shot_score)
    )
    authors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authors)
    import pandas as pd

    loader = authors.create_dataloader(
        contexts, tokenizer, batch_size, target_index, stepSize=1, useMasking=True
    )
    probs = authors.extract_logits(model, loader, device, target_index, tokenizer, stepSize=1)
    return authors.zero_shot_score(pd.DataFrame(variants), probs)
