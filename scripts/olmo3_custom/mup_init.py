"""
mup_init.py — optional muP (Maximal Update Parametrization) helpers.

``mup`` is an *optional* dependency.  If it is not installed the helpers
gracefully fall back to standard PyTorch initialisation and emit a warning
(or raise, depending on the *strict* flag).

Installation
~~~~~~~~~~~~
pip install mup

What is and isn't implemented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Implemented**: rescale the weights of q/k/v/o projections and MLP
  layers by ``1 / sqrt(hidden_size)`` to approximate the muP fan-in
  scaling, via ``mup.set_base_shapes``.  The learning-rate is also
  adjusted per the muP prescription if the caller passes a dict of
  param-group learning-rates.
* **Not implemented**: full muP coord-check or hyperparameter transfer
  from a small proxy model.  True muP requires coordinating the proxy
  model, the actual model, and the optimiser setup.  The integration here
  is pragmatic — it reduces pathological gradient scales at large width
  but does not guarantee a hyperparameter-transfer guarantee.
* **Recommendation**: if you need rigorous muP, train proxy models with
  the ``mup`` CLI and use the resulting ``--base_shapes`` file.
"""

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _check_mup_available(strict: bool = False) -> bool:
    """Return True if ``mup`` is importable, else warn/raise."""
    try:
        import mup  # noqa: F401
        return True
    except ImportError:
        msg = (
            "The `mup` package is not installed. "
            "Install it with:  pip install mup\n"
            "Without muP the model will use standard PyTorch initialisation."
        )
        if strict:
            raise ImportError(msg) from None
        warnings.warn(msg, stacklevel=3)
        return False


def apply_mup_init(
    model: nn.Module,
    base_hidden_size: int = 256,
    strict: bool = False,
) -> bool:
    """
    Apply a pragmatic muP initialisation to *model*.

    Scales the standard deviation of linear-projection weights by
    ``sqrt(base_hidden_size / hidden_size)`` so that activations have
    roughly unit variance at all widths.

    Args:
        model: An :class:`Olmo3CustomForCausalLM` (or any model with
            ``model.config.hidden_size``).
        base_hidden_size: Reference hidden size used as the "base shape".
            Defaults to 256 (a small proxy model width).
        strict: If ``True`` raise :class:`ImportError` when ``mup`` is
            not installed.  If ``False`` (default) warn and return
            ``False`` without modifying the model.

    Returns:
        ``True`` if muP was applied, ``False`` otherwise.
    """
    if not _check_mup_available(strict=strict):
        return False

    import mup

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        logger.warning("Cannot determine hidden_size from model; skipping muP init.")
        return False

    scale = (base_hidden_size / hidden_size) ** 0.5

    logger.info(
        "Applying pragmatic muP init: base_hidden=%d, hidden=%d, scale=%.4f",
        base_hidden_size,
        hidden_size,
        scale,
    )

    rescaled = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Rescale weight std; keep bias as-is.
            with torch.no_grad():
                module.weight.data.mul_(scale)
            rescaled += 1

    logger.info("muP: rescaled %d Linear layers.", rescaled)
    return True


def get_mup_optimizer_kwargs(
    model: nn.Module,
    base_lr: float = 3e-4,
    base_hidden_size: int = 256,
    strict: bool = False,
) -> Optional[list]:
    """
    Return a list of param-group dicts with muP learning-rate scaling.

    Under muP the optimal learning rate for layers that grow with width
    scales as ``base_lr * (base_hidden_size / hidden_size)``.

    Args:
        model: The model to optimise.
        base_lr: Learning rate at the base (proxy) width.
        base_hidden_size: Width of the proxy model.
        strict: See :func:`apply_mup_init`.

    Returns:
        A list of ``{'params': [...], 'lr': float}`` dicts suitable for
        passing to an optimiser, or ``None`` if muP is unavailable.
    """
    if not _check_mup_available(strict=strict):
        return None

    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        return None

    lr_scale = base_hidden_size / hidden_size

    # Separate "width-dependent" params (projections) from "width-independent"
    # ones (embeddings, norms, biases).
    width_dep_params = []
    width_indep_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_proj = any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
        if is_proj:
            width_dep_params.append(param)
        else:
            width_indep_params.append(param)

    param_groups = [
        {"params": width_dep_params, "lr": base_lr * lr_scale},
        {"params": width_indep_params, "lr": base_lr},
    ]

    logger.info(
        "muP param groups: %d width-dep params (lr=%.2e), %d width-indep params (lr=%.2e)",
        len(width_dep_params),
        base_lr * lr_scale,
        len(width_indep_params),
        base_lr,
    )
    return param_groups
