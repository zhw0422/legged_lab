# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0",
    entry_point="legged_rl_lab.envs:AMPManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:UnitreeG1AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-skrl-v0",
    entry_point="legged_rl_lab.envs:AMPManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:UnitreeG1AMPFlatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_ppo_cfg.yaml",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0",
    entry_point="legged_rl_lab.envs:AMPManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:UnitreeG1AMPFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeG1AMPFlatPPORunnerCfg",
    },
)
