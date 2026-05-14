# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied locomotion tasks.

Tasks for training a single policy across multiple robot embodiments
(e.g., G1 + Go2, procedural quadrupeds, procedural humanoids).
"""

# Register ActorCriticWithEncoder into rsl_rl.modules so that the RSL-RL
# runner can resolve it via eval("rsl_rl.modules.ActorCriticWithEncoder").
# If an alternative rsl_rl (e.g. the custom AMP one) is on sys.path and
# doesn't expose ActorCritic, skip registration — the cross-embodied tasks
# will still import but those specific Gym IDs won't be usable with the
# alternative rsl_rl.  The AMP training script only needs AMP task IDs.
from .mdp.cross_procedural_mdp import register_in_rsl_rl as _register_encoder  # noqa
from .mdp.cross_procedural_mdp import CrossEmbodiedEncoderCfg  # noqa: F401
try:
    _register_encoder()
except ImportError as _e:
    import warnings
    warnings.warn(
        f"[cross_emboided] Skipping ActorCriticWithEncoder registration — "
        f"active rsl_rl has no ActorCritic ({_e}).  Cross-embodied tasks will "
        f"import but cannot be trained with this rsl_rl build.",
        stacklevel=1,
    )

from .config.g1go2_mixed import *  # noqa
from .config.procedural_quadruped import *  # noqa
from .config.procedural_humanoid import *  # noqa
from .config.procedural_mixed import *  # noqa
