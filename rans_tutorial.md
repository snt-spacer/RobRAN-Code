# RANS Tutorial: Designing a Robot System and Training an RL Agent

This tutorial shows hot to create a robotic system, linking it to tasks, and training a Reinforcement Learning (RL) agent using the RANS version of Isaac Lab framework. As an example, we use the robot named `floating_platform`.

![](RANS_logo.png)
---

## Overview of the Example Files and Structure

### Main Folders and Their Roles
- **`IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/robots/`**:
  Contains the robot configuration and class descriptions. For example, `floating_platform.py` specifies the `floating_platform` robot's properties.

- **`IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ROBOT_NAME`**:
  Contains task descriptions for robot-task pairs. Tasks are registered as Gym environments with unique IDs in the `__init__.py` file.

- **Subdirectories of `rans`**:
  - `robots`: Contains specific robot class definitions.
  - `tasks`: Implements various tasks for the robots, like navigation or velocity tracking.
  - `tasks_cfg`: Contains task-specific configuration files.
  - `utils`: Provides utility functions like generation of unique per-environment RNG, functionalities to evaluate and plot the results of testing a trained model for a specific robot-task pair.

---

## Step 1: Configuring the Robot

### Example File: `floating_platform.py`
**Location**: `IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/robots/floating_platform.py`

This file defines the robot configuration using the `FLOATING_PLATFORM_CFG` object. Key aspects include:
- **USD Path**: Specifies the 3D model file.
- **Physical Properties**: Enables simulation features like gravity and velocity constraints.
- **Initial State**: Sets default positions and joint values.

Example:
```python
FLOATING_PLATFORM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REPO_ROOT_PATH}/floating_platform.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
)
```

---

## Step 2: Defining Tasks

### Example Folder: `tasks/`
**Location**: `IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/tasks/`

Tasks are implemented here. For example:
- **File**: `go_to_position.py` describes the "Go To Position" task.
- **Core Task Logic**: Uses the `TaskCore` base class to define observation space, rewards, and task dynamics.

Example:
```python
class GoToPositionTask(TaskCore):
    def get_observations(self):
        # Observation logic
        pass

    def compute_rewards(self):
        # Reward logic
        pass
```

---

## Step 3: Configuring Task Environments

### Example File: `floating_platform_go_to_position.py`
**Location**: `IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/floating_platform_go_to_position.py`

This file links the robot and task configurations to create an RL environment.

Example:
```python
@configclass
class FloatingPlatformGoToPositionEnvCfg(DirectRLEnvCfg):
    scene = InteractiveSceneCfg(num_envs=4096)
    robot_cfg = FloatingPlatformRobotCfg()
    task_cfg = GoToPositionCfg()
    episode_length_s = 40.0
```

---

## Step 4: Registering Gym Environments

### Example File: `__init__.py`
**Location**: `IsaacLab_RANS/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/floating_platform/__init__.py`

Gym environments are registered with unique IDs, linking them to specific task and robot configurations.

Example registration:
```python
gym.register(
    id="Isaac-FloatingPlatform-GoToPosition-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.floating_platform:FloatingPlatformGoToPositionEnv",
    kwargs={
        "env_cfg_entry_point": FloatingPlatformGoToPositionEnvCfg,
    },
)
```

---

## Step 5: Training an RL Agent

### RL Libraries
Compatible libraries include `rl_games`, `skrl`, and `Stable-Baselines3`. Example configuration files are referenced in the `__init__.py`.

### Training Command
Use the environment ID (e.g., `"Isaac-FloatingPlatform-GoToPosition-Direct-v0"`) to start training.

Example command for `rl_games`:
```bash
 ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-FloatingPlatform-GoToPosition-Direct-v0 --num_envs 4096 --headless
```

Alternatively, to easily switch between robot and tasks, one can use the following:
```bash
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-RANS-Single-v0 --num_envs 4096 --headless env.robot_name=FloatingPlatform env.task_name=GoToPosition
```

In the previous command, the `env.robot_name` is used to select which robot should be used, and the `env.task_name` is used to select the task that should be loaded.
These names relate to the ones given inside the factories in these files:
```bash
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/robots_cfg/__init__.py
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/robots/__init__.py
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/tasks_cfg/__init__.py
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/rans/tasks/__init__.py
```

---
