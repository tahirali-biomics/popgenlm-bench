"""Optional PlantCAD scoring utilities; model dependencies load only at runtime."""
from __future__ import annotations
import math

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMP)[::-1].upper()

def coordinate_to_context(sequence: str, position: int, context_size: int = 512, target_index: int = 255, pad: str = "N") -> tuple[str, int]:
    if position < 1 or context_size <= 0 or not 0 <= target_index < context_size:
        raise ValueError("invalid coordinate or context parameters")
    zero = position - 1
    start = zero - target_index
    end = start + context_size
    left = max(0, -start); right = max(0, end - len(sequence))
    context = pad * left + sequence[max(0, start):min(len(sequence), end)] + pad * right
    if len(context) != context_size:
        raise AssertionError("context length mismatch")
    return context.upper(), target_index

def validate_reference(sequence: str, position: int, ref: str) -> None:
    if position < 1 or position > len(sequence):
        raise ValueError("position out of bounds")
    observed = sequence[position - 1].upper()
    if observed != ref.upper():
        raise ValueError(f"REF mismatch at {position}: expected {ref}, observed {observed}")

def raw_score(log_probs, ref: str, alt: str, allele_index: dict[str, int] | None = None) -> float:
    idx = allele_index or {"A": 0, "C": 1, "G": 2, "T": 3}
    value = float(log_probs[idx[alt.upper()]]) - float(log_probs[idx[ref.upper()]])
    if not math.isfinite(value): raise ValueError("non-finite score")
    return value

def harmonized_score(raw: float) -> float:
    value = -float(raw)
    if not math.isfinite(value): raise ValueError("non-finite score")
    return value

def load_local_model(checkpoint: str, device: str = "cuda", revision: str | None = None):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    kwargs = {"trust_remote_code": True, "local_files_only": True, "torch_dtype": torch.float32}
    if revision: kwargs["revision"] = revision
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs).to(device=device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model.eval()
    return model, tokenizer

def comparison_pass(observed: float, expected: float, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    if not (math.isfinite(observed) and math.isfinite(expected)):
        return False
    return abs(observed-expected) <= atol + rtol*abs(expected)

def validate_comparison_ids(rows, expected_ids):
    ids=[r['variant_id'] for r in rows]
    if len(ids)!=len(expected_ids) or len(set(ids))!=len(ids) or set(ids)!=set(expected_ids):
        raise ValueError('missing, duplicate, or unexpected variant IDs')
    for r in rows:
        for k in ('adapter_batch1','authors_reference','adapter_batch8','reverse_complement'):
            if not math.isfinite(float(r[k])): raise ValueError('nonfinite comparison score')

def population_scores(ref, alt, af_alt, score):
    """Reuse frozen v0.2 rules, including undefined minor/major at frequency ties."""
    from .orientation import orient_gpn_score
    oriented = orient_gpn_score(ref, alt, af_alt, score)
    minor = oriented["gpn_score_minor_vs_major"]
    return {
        "plantcad_score_ref_alt": float(score),
        "plantcad_score_ref_alt_negative": harmonized_score(score),
        "plantcad_score_paper_oriented": oriented["gpn_score_paper_oriented"],
        "plantcad_score_minor_vs_major": minor,
        "plantcad_score_minor_vs_major_negative": None if minor is None else -minor,
        "minor_allele": oriented["minor_allele"],
        "major_allele": oriented["major_allele"],
        "orientation": oriented["orientation"],
        "flipped": oriented["flipped"],
    }


def prepare_masked_inputs(tokenizer, chunk, batch_variants, target_index=255):
    """Tokenize and mask on CPU; accept DNA IUPAC symbols without shifting bases."""
    import torch

    # Pinned tokenizer emits one [UNK]=2 per noncanonical IUPAC base.
    # Validate the original symbols and every token ID; decoding is lossy.
    vocab = tokenizer.get_vocab()
    ids = {base: vocab[base.lower()] for base in 'ACGT'}
    if ids != dict(A=3, C=4, G=5, T=6) or tokenizer.mask_token_id != 1:
        raise ValueError('unexpected pinned tokenizer vocabulary')
    x = tokenizer(chunk, add_special_tokens=False, return_tensors='pt')['input_ids']
    if tuple(x.shape) != (len(chunk), 512):
        raise ValueError('expected one token per base in 512-base context')
    for i, (context, variant) in enumerate(zip(chunk, batch_variants)):
        if len(context) != 512 or context[target_index] != variant['ref']:
            raise ValueError('REF/target alignment mismatch')
        expected = torch.tensor([ids[b] if b in ids else 2 for b in context])
        if set(context) - set('ACGTRYSWKMBDHVN') or not torch.equal(x[i].cpu(), expected):
            raise ValueError('base-to-token alignment mismatch')
    before = x.clone()
    x[:, target_index] = tokenizer.mask_token_id
    changed = x != before
    if not (torch.all(changed.sum(dim=1) == 1) and torch.all(changed[:, target_index]) and torch.all((x == tokenizer.mask_token_id).sum(dim=1) == 1)):
        raise ValueError('exactly the target must be masked')
    return x

def score_contexts(model, tokenizer, contexts, variants, batch_size=8, target_index=255):
    """Yield masked scores using the float32 scoring convention.

    Imports are lazy. Callers must supply an eval-mode CUDA float32 model.
    Contexts are reference-forward; no strand averaging is performed.
    """
    import hashlib
    import torch

    if len(contexts) != len(variants) or batch_size < 1:
        raise ValueError("context/variant counts or batch size invalid")
    if model.training or next(model.parameters()).dtype != torch.float32:
        raise ValueError("float32 evaluation model required")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("CUDA required; no CPU fallback")
    vocab = tokenizer.get_vocab()
    ids = {base: vocab[base.lower()] for base in "ACGT"}
    if ids != dict(A=3, C=4, G=5, T=6) or tokenizer.mask_token_id != 1:
        raise ValueError("unexpected pinned tokenizer vocabulary")
    for start in range(0, len(contexts), batch_size):
        chunk = contexts[start:start + batch_size]
        batch_variants = variants[start:start + batch_size]
        x = prepare_masked_inputs(tokenizer, chunk, batch_variants, target_index)
        with torch.inference_mode():
            logits = model(x.to(device)).logits[:, target_index, :].float()
            nuc = logits[:, [ids[b] for b in "ACGT"]]
            logp = torch.log_softmax(nuc, dim=1)
            probs = torch.softmax(nuc, dim=1)
        if not (torch.isfinite(logits).all() and torch.isfinite(logp).all()
                and torch.all(torch.abs(probs.sum(dim=1) - 1) < 1e-6)):
            raise ValueError("nonfinite logits or invalid A/C/G/T normalization")
        lp, nl = logp.cpu().tolist(), nuc.cpu().tolist()
        for i, variant in enumerate(batch_variants):
            ri, ai = "ACGT".index(variant["ref"]), "ACGT".index(variant["alt"])
            raw = lp[i][ai] - lp[i][ri]
            if not math.isfinite(raw) or abs(raw - (nl[i][ai] - nl[i][ri])) >= 1e-5:
                raise ValueError("nonfinite score or logit/score identity mismatch")
            yield {
                **population_scores(variant["ref"], variant["alt"], variant["af_alt"], raw),
                "ref_logprob": lp[i][ri], "alt_logprob": lp[i][ai],
                "context_sha256": hashlib.sha256(chunk[i].encode()).hexdigest(),
                "masked_input_ids_sha256": hashlib.sha256(x[i].numpy().tobytes()).hexdigest(),
            }
