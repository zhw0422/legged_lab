# legged_rl_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


## 🧰️ Setup 

* Use pip to install isaaclab [pip install isaaclab](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html)


* Create conda environment
```bash
conda create -n legged_rl_lab python=3.11
conda activate legged_rl_lab
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
```

* Install isaacsim 5.1 and isaaclab 2.3
```bash
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```

* Verify the installization
```bash
isaacsim
```

* Install the project
```bash
python -m pip install -e source/legged_rl_lab
python -m pip install -e source/legged_rl_lab/legged_rl_lab/rsl_rl
```

* List the tasks available in the project
```bash
python scripts/list_envs.py
```

---

## 🚀 Train

### 🐕️ Go2

<details>
<summary><b>Walk (Flat)</b></summary>

#### Walk (Flat)

[<img src="media/walkflat_isaac.gif" width="300px">](gifs/isaac.gif)


```bash 
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go1-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run /path/to/log/folder \
  --checkpoint model_xx.pt  
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 16
```


</details>

<details>
<summary><b>Walk (Rough)</b></summary>

#### Walk(rough)

[<img src="media/walkrough_isaac.gif" width="300px">](gifs/walkrough.gif)

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 4096 \
  --headless

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=54321 \
    scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 4096 \
    --headless  
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0 \
    --num_envs 16
```


</details>

<details>
<summary><b>Handstand</b></summary>

#### Footstand

[<img src="media/footstand_isaac.gif" width="300px">](gifs/isaac.gif)


```bash
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Footstand-Unitree-Go2-v0 \
  --num_envs 4096 \
  --headless
```

```bash
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Handstand-Unitree-Go2-v0 \
    --num_envs 16
```


</details>

### 🤖️ G1

<details>
<summary><b>Walk (Flat)</b></summary>

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
    --num_envs 16
```

</details>

<details>
<summary><b>Walk (Rough)</b></summary>

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 16
```

</details>

<details>
<summary><b>Cross-Embodied G1+Go2 (Mixed)</b></summary>

```bash
#Train (multi-GPU)
python -m torch.distributed.run \
  --nproc_per_node=4 \
  scripts/rsl_rl/train_cross_embodied_shared.py \
  --num_envs 4096 \
  --headless

#Train (single-GPU)
python scripts/rsl_rl/train_cross_embodied_shared.py \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play_cross_embodied_shared.py \
  --num_envs 32
```

</details>

<details>
<summary><b>Procedural Quadruped</b></summary>

```bash
# Flat – Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Quadruped-v0 \
    --num_envs 4096 \
    --headless

# Flat – Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Quadruped-Play-v0 \
    --num_envs 32

# Rough – Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Quadruped-v0 \
    --num_envs 4096 \
    --headless

# Rough – Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Quadruped-Play-v0 \
    --num_envs 32
```

</details>

<details>
<summary><b>Procedural Humanoid</b></summary>

```bash
# Flat – Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Humanoid-v0 \
    --num_envs 4096 \
    --headless

# Flat – Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Humanoid-Play-v0 \
    --num_envs 32

# Rough – Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Humanoid-v0 \
    --num_envs 4096 \
    --headless

# Rough – Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Humanoid-Play-v0 \
    --num_envs 32
```

</details>

<details>
<summary><b>Procedural Mixed (Humanoid + Quadruped)</b></summary>

Trains a **single policy** across procedurally generated bipeds and quadrupeds simultaneously.
Three pluggable obs-encoder back-ends are available:

| Encoder | Flat Train Task | Rough Train Task |
|---------|----------------|-----------------|
| Mask (default) | `…-Flat-Procedural-Mixed-v0` | `…-Rough-Procedural-Mixed-v0` |
| Transformer | `…-Flat-Procedural-Mixed-Transformer-v0` | `…-Rough-Procedural-Mixed-Transformer-v0` |
| GCN | `…-Flat-Procedural-Mixed-GCN-v0` | `…-Rough-Procedural-Mixed-GCN-v0` |

Architecture note: encoder lives in `mdp/cross_procedural_mdp.py`; all three procedural env
types (`ProceduralHumanoidRobotEnv`, `ProceduralQuadrupedRobotEnv`, `ProceduralMixedRobotEnv`)
inherit from `CrossProceduralEnv` which provides the unified morphology-params interface.

```bash
# ── Flat – Mask (default) ────────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-v0 \
    --num_envs 4096 \
    --headless

# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-Play-v0 \
    --num_envs 32

# ── Flat – Transformer ───────────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-Transformer-v0 \
    --num_envs 4096 \
    --headless

# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-Play-v0 \
    --num_envs 32

# ── Flat – GCN ───────────────────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-GCN-v0 \
    --num_envs 4096 \
    --headless

# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-GCN-Play-v0 \
    --num_envs 32

# ── Rough – Mask (default) ───────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-v0 \
    --num_envs 4096 \
    --headless

# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-Play-v0 \
    --num_envs 32

# ── Rough – Transformer ──────────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-Transformer-v0 \
    --num_envs 4096 \
    --headless

# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-Play-v0 \
    --num_envs 32

# ── Rough – GCN ──────────────────────────────────────────────────────────
# Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-GCN-v0 \
    --num_envs 4096 \
    --headless
    
# Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-GCN-Play-v0 \
    --num_envs 32
```

</details>

### 🧗 Parkour

<details>
<summary><b>Depth</b></summary>

#### TS-Depth Teacher

```bash
# G1 — phase 1 teacher / privileged-latent training
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-v0 \
  --num_envs 4096 \
  --headless

# G1 — teacher play
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-Play-v0 \
  --num_envs 50
```

```bash
# Go2 — phase 1 teacher / privileged-latent training
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-v0 \
  --num_envs 4096 \
  --headless

# Go2 — teacher play
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-Play-v0 \
  --num_envs 50
```

#### TS-Depth Student

Use the phase-1 checkpoint as the teacher source. With `--resume`, the distill config automatically uses that checkpoint as `algorithm.teacher_checkpoint_path`.

```bash
# G1 — phase 2 student distillation
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-Distill-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <teacher_run_folder> \
  --checkpoint model_xxx.pt

# Go2 — phase 2 student distillation
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-Distill-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <run_folder> \
  --checkpoint model_xxx.pt

# Export the student depth policy from the distillation run
python scripts/rsl_rl/export_ts_depth_policy.py \
  --checkpoint logs/rsl_rl/go2_parkour_depth_distill/<student_run_folder>/model_xxx.pt \
  --onnx
```

</details>

<details>
<summary><b>Attention</b></summary>

#### AME Attention — Overview

CNN + Multi-Head Attention encoder over a 33×21×3 local terrain map (1.6m × 1.0m at 0.05m resolution).

- **CNN**: 2-layer (3→16→64), stride=2 downsampling
- **MHA**: 16 heads, embed_dim=64, Query=proprioceptive embedding, Key/Value=CNN features
- **Actor/Critic MLP**: [512, 256, 128], ELU activation
- **PPO**: `entropy_coef=0.008`, `init_noise_std=1.0`, `lr=1e-3` adaptive schedule

---

#### Train — 10×20 curriculum terrain

8 terrain types matching AME `ROUGH_TERRAINS_CFG`: stairs up/down (10%+10%), boxes (10%), random rough (10%), slopes (10%+10%), stepping stones (20%), concentric gaps (20%).

```bash
# Train from scratch
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-v0 \
  --num_envs 2048 \
  --headless \
  --logger tensorboard

# Resume from checkpoint
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-v0 \
  --num_envs 2048 \
  --headless \
  --logger tensorboard \
  --resume \
  --load_run 2026-06-18_18-15-43 \
  --checkpoint model_57500.pt \
  --run_name my_run

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=54321 \
  scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-v0 \
  --num_envs 4096 --headless --logger tensorboard
```

| CLI arg | Description | Default |
|---------|-------------|---------|
| `--num_envs` | Parallel envs | 512 |
| `--max_iterations` | Max training iters | 100000 |
| `--logger` | `wandb`, `tensorboard`, or `neptune` | `wandb` |
| `--resume` | Resume from checkpoint | off |
| `--load_run` | Run folder name to resume from | — |
| `--checkpoint` | Checkpoint file name | — |
| `--run_name` | Suffix for the new run folder | — |

> **Note:** Use `--logger tensorboard` in headless mode without a wandb API key.

---

#### Finetune — Stake-focused training

Set `FINETUNE = True` in `g1_attention_env_cfg.py`. Matches AME `FINETUNE_ROUGH_TERRAINS_CFG` exactly.

**1. Terrain mix (8 types, 100% total):**

| Terrain | Prop. | Description |
|---------|-------|-------------|
| pyramid_stairs | 10% | Stairs up (0.05–0.25m) |
| pyramid_stairs_inv | 10% | Stairs down (0.05–0.25m) |
| stakes1 (double) | 10% | Double column, fixed gap 0.1m |
| stakes2 (alternate) | 20% | Alternate column, gap 0.0–0.2m |
| stakes3 (alternate) | 20% | Alternate column, gap 0.3–0.2m |
| hf_gaps | 10% | Concentric gaps (0.2–0.6m) |
| stones_bridge | 10% | Stone bridge |
| rails | 10% | Rails (0.25→0.05m) |

**2. Reward weight changes:**

| Reward | Regular | Finetune | Change |
|--------|---------|----------|--------|
| tracking_lin_vel | 2.0 | 2.0 | — |
| dof_torques_limits | -0.01 | **-0.05** | ×5 |
| action_rate_l2 | -0.01 | **-0.05** | ×5 |
| flat_orientation | -2.0 | **-5.0** | ×2.5 |
| feet_air_time | 0.25 | **0.5** | ×2 |
| feet_air_time_variance | -0.1 | **-2.0** | ×20 |
| feet_slide | -0.1 | **-0.3** | ×3 |
| feet_stumble | -1.0 | **-5.0** | ×5 |
| feet_too_near | -1.0 | **-5.0** | ×5 |
| joint_coordination | -0.2 | **-0.5** | ×2.5 |

**3. Randomization disabled:**
- `push_robot = None`, `add_base_mass = None`, `base_com = None`
- Observation noise off (`enable_corruption = False`)
- Fixed spawn position, fixed heading = (0, 0)

```bash
# 1. Set FINETUNE = True in g1_attention_env_cfg.py
# 2. Start finetune from a pretrained checkpoint
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-v0 \
  --num_envs 2048 \
  --headless \
  --logger tensorboard \
  --resume \
  --load_run 2026-06-18_18-15-43 \
  --checkpoint model_57500.pt \
  --run_name finetune_stones

# 3. Set FINETUNE = False after finetune is done
```

---

#### Play — Evaluation & attention debugging

```bash
# Basic play
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-Play-v0 \
  --num_envs 14

# Load specific checkpoint
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-Play-v0 \
  --num_envs 14 \
  --ckpt model_54000.pt

# Attention viz + stats + save weights
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-Play-v0 \
  --num_envs 14 \
  --ckpt model_54000.pt \
  --vis_attention \
  --print_attention_stats \
  --save_attention_weights

# Headless eval
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-Play-v0 \
  --num_envs 14 \
  --ckpt model_54000.pt \
  --headless
```

**Attention play flags:**

| Flag | Description |
|------|-------------|
| `--vis_attention` | Colored spheres on terrain (red=high attn, blue=low) |
| `--print_attention_stats` | Print per-step stats (mean, entropy, spatial L/R/F/B) |
| `--attention_print_interval` | Stats print interval in steps (default 50) |
| `--save_attention_weights` | Save `.npy` for heatmap GIF generation |
| `--ckpt` | Checkpoint file name under `ckpt/` |
| `--headless` | Run without GUI |

**Play terrain presets** (in `configure_g1_attention_play_terrain()`):

| Preset | Terrain | Use case |
|--------|---------|----------|
| A | AlternateColumnStakes | Stakes (default) |
| B | DoubleColumnStakes | Double stakes |
| C | SteppingStones | Stepping stones |
| D | StonesBridge | Stone bridge |
| E | ConcentricGaps | Concentric gaps |
| F | PyramidStairs (up) | Stairs up |
| G | PyramidStairs (down) | Stairs down |

---

#### Go2

```bash
# Go2 — train
python scripts/rsl_rl/train.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-Go2-v0 \
  --num_envs 2048 \
  --headless \
  --logger tensorboard

# Go2 — play
python scripts/rsl_rl/play.py \
  --task LeggedRLLab-Isaac-Parkour-Attention-Unitree-Go2-Play-v0 \
  --num_envs 50
```

</details>

### 🏃 Mimic

#### Datasets

Place motion datasets under the project motion directory:

```
source/legged_rl_lab/legged_rl_lab/data/motion/
├── LAFAN1_Retargeting_Dataset/   # Motion capture retargeted CSV (30 FPS)
│   └── g1/                       # 40 CSV clips (walk, run, dance, jump, fight, fall …)
└── AMASS_Retargeted_for_G1/      # Large-scale motion capture NPZ (25 sub-libraries, 17,714 files)
    └── g1/
        ├── CMU/
        ├── KIT/
        └── ...
```

- LAFAN1 retargeted data: [LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- AMASS retargeted data: [AMASS_Retargeted_for_G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)

LAFAN1 is stored as retargeted `.csv`, so convert it to `.npz` before AMP or tracking training.

```bash
# Convert one CSV.
python scripts/csv_to_npz.py \
  --input_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv \
  --input_fps 30 \
  --output_fps 30 \
  --headless
```

```bash
# Or batch-convert every CSV under the G1 LAFAN1 folder.
# Remove --overwrite if you want to keep existing NPZ files.
python scripts/csv_to_npz.py \
  --input_dir source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1 \
  --input_fps 30 \
  --output_fps 30 \
  --headless \
  --overwrite
```

For AMP locomotion training, put the converted LAFAN1 `run*.npz` and `walk*.npz` files into `source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_amp_run_walk_fall_getup/`.

```bash
# Optional: replay one converted NPZ in Isaac Sim to verify the body state.
python scripts/replay_npz.py \
  --file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_amp_run_walk_fall_getup/walk1_subject1.npz
```


<details>
<summary><b>AMP</b></summary>

```bash
# Train — G1 humanoid, flat terrain, AMP + RSI
# Default expert motion: LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless

# Train on one specific motion file
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.npz

# Train on a directory of motions
# Use the folder containing converted LAFAN1 run/walk NPZ files.
python scripts/amp/train.py \
  --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
  --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_amp_run_walk_fall_getup \
  --num_envs 4096 \
  --headless \
  --max_iterations 20000

# Resume from a checkpoint
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless \
    --resume
```

```bash
# Play / visualise
python scripts/amp/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0 \
    --num_envs 50 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_amp_run_walk_fall_getup
```

**skrl AMP** (alternative AMP implementation with 3-way discriminator loss):

```bash
# Train — G1 humanoid, flat terrain, skrl AMP
python scripts/skrl/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-skrl-v0 \
    --algorithm AMP \
    --num_envs 4096 \
    --headless \
    --max_iterations 20000
```

```bash
# Play — auto-loads the latest checkpoint under
# logs/skrl/unitree_g1_amp_flat_skrl/<run>/checkpoints/
python scripts/skrl/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-skrl-v0 \
    --algorithm AMP \
    --num_envs 50

# Play — load a specific checkpoint by absolute path
python scripts/skrl/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-skrl-v0 \
    --algorithm AMP \
    --num_envs 50 \
    --checkpoint logs/skrl/unitree_g1_amp_flat_skrl/<run_folder>/checkpoints/agent_24000.pt

# Play — load a checkpoint copied to the local ckpt/ directory
python scripts/skrl/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-skrl-v0 \
    --algorithm AMP \
    --num_envs 50 \
    --ckpt agent_24000.pt
```

</details>



<details>
<summary><b>Motion Tracking</b></summary>

[<img src="media/mimic_lafan.gif" width="300px">](gifs/walkrough.gif)

| Task ID | Description |
|---------|-------------|
| `Tracking-Flat-G1-v0` | Standard, with state estimation |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | No state estimation (closer to real deployment) |
| `Tracking-Flat-G1-Low-Freq-v0` | Half-frequency control |



```bash
# Step 3 — Train
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file </path/to/npz/file> \
  --num_envs 4096 --headless

# Resume
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file <path/to/npz/file> \
  --resume --load_run <run_folder> --checkpoint model_xxx.pt \
  --num_envs 4096 --headless
```

```bash
# Step 4 — Play
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /path/to/motion.npz \
  --num_envs 16

python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_jump/jumps1_subject1.npz \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/g1_flat/2026-04-02_02-32-52/model_11000.pt

```

</details>


---

## Sim2Sim

Terrain Generator: use the terrain generator script, see [terrain_tool](deploy/utils/terrain_tool/readme.md) for details.

```bash
python3 deploy/utils/terrain_tool/terrain_generator.py
```

<details>
<summary><b>Go1 Walk</b></summary>

See [deploy/go1_deploy/README.md](deploy/go1_deploy/README.md) for details.

```bash
pip install mujoco
python deploy/go1_deploy/sim2sim_walk.py --model go1_flat.pt
```

</details>

<details>
<summary><b>Go2 Walk / Handstand</b></summary>

See [deploy/go2_deploy/README.md](deploy/go2_deploy/README.md) for details.

```bash
pip install mujoco
# Walk
python deploy/go2_deploy/sim2sim_walk.py --model go2_rough.pt
# Handstand
python deploy/go2_deploy/sim2sim_handstand.py --model go2_handstand.pt
```

</details>

<details>
<summary><b>G1 Walk</b></summary>

See [deploy/g1_deploy/README.md](deploy/g1_deploy/README.md) for details.

```bash
pip install mujoco
python deploy/g1_deploy/sim2sim_walk.py --model g1_flat_1.onnx --config g1_walk.yaml
```

</details>

---

## Sim2Real

<details>
<summary><b>Go1 Walk</b></summary>

See [deploy/go1_deploy/README.md](deploy/go1_deploy/README.md) for details.

```bash
# Dependency: unitree_legged_sdk (see README)
python deploy/go1_deploy/sim2real_walk.py --mode real --model policy.pt
```

</details>

<details>
<summary><b>Go2 Walk</b></summary>

See [deploy/go2_deploy/README.md](deploy/go2_deploy/README.md) for details.

```bash
python deploy/go2_deploy/sim2real_walk.py --mode real --model policy.pt
```

</details>

<details>
<summary><b>G1 Walk</b></summary>

See [deploy/g1_deploy/README.md](deploy/g1_deploy/README.md) for details.

```bash
# Dependency: cyclonedds + unitree_sdk2_python (see README)
python deploy/g1_deploy/sim2real_walk.py
```

</details>



## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/legged_rl_lab"
    ]
}
```

### Restart Terminal
```bash
pkill -f "python.*train.py"
```


## Acknowledgements

* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file#acknowledgements)
* [legged_lab](https://github.com/zitongbai/legged_lab)
* [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
* [beyondmimic](https://github.com/HybridRobotics/whole_body_tracking)
