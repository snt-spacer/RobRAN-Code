# Domain Randomization

This module provides a comprehensive set of randomization techniques for robust policy learning. Domain randomization helps improve the generalization capabilities of learned policies by exposing them to varied environmental conditions during training.

## Overview

The randomization system is built around a core architecture that supports different timing modes:
- **Reset-based**: Applied when environments reset (episode start)
- **Step-based**: Applied every simulation step (during episode)
- **Action-based**: Applied when processing actions
- **Observation-based**: Applied when processing observations

## Randomization Timing Table

The following table shows all available randomization types, their modes, and when each mode is applied:

| Randomization Type | Mode Name | Applied At | Description |
|-------------------|-----------|------------|-------------|
| **Mass** | `"uniform"` | Reset | Uniform mass variation on episode start |
| **Mass** | `"normal"` | Reset | Normal distribution mass variation on episode start |
| **Mass** | `"constant_time_decay"` | Step | Mass decays over time during episode |
| **Mass** | `"action_based_decay"` | Step | Mass decays based on action magnitude |
| **CoM** | `"uniform"` | Reset | Uniform CoM position variation on episode start |
| **CoM** | `"normal"` | Reset | Normal distribution CoM variation on episode start |
| **CoM** | `"spring"` | Step | Spring-like CoM behavior (not implemented) |
| **Inertia** | `"uniform"` | Reset | Uniform inertia variation on episode start |
| **Inertia** | `"normal"` | Reset | Normal distribution inertia variation on episode start |
| **Inertia** | `"decay"` | Step | Inertia decays over time during episode |
| **Noisy Actions** | `"uniform"` | Reset + Action | Uniform noise sampled on reset, applied during action processing |
| **Noisy Actions** | `"normal"` | Reset + Action | Normal noise sampled on reset, applied during action processing |
| **Actions Rescaler** | `"uniform"` | Reset + Action | Uniform scaling factors sampled on reset, applied during action processing |
| **Noisy Observations** | `"uniform"` | Reset + Observation | Uniform noise sampled on reset, applied during observation processing |
| **Noisy Observations** | `"normal"` | Reset + Observation | Normal noise sampled on reset, applied during observation processing |
| **Wrench** | `"kick_uniform"` | Reset + Step | Sporadic kicks with uniform distribution, timing controlled during steps |
| **Wrench** | `"kick_normal"` | Reset + Step | Sporadic kicks with normal distribution, timing controlled during steps |
| **Wrench** | `"constant_uniform"` | Reset + Step | Constant wrenches with uniform distribution, applied every step |
| **Wrench** | `"constant_normal"` | Reset + Step | Constant wrenches with normal distribution, applied every step |
| **Wrench** | `"constant_sinusoidal"` | Reset + Step | Sinusoidal varying wrenches, applied every step |

**Notes:**
- **Reset**: Applied once when the environment resets (episode start)
- **Step**: Applied every simulation step during the episode
- **Action**: Applied when processing robot actions
- **Observation**: Applied when processing observations

## Types of Randomization

### 1. **Mass Randomization** (`MassRandomizationCfg`)

**Purpose**: Randomizes the mass of robot bodies to simulate variations in payload, wear, or manufacturing differences.

**Available Modes**:
- `"uniform"` - Uniform distribution on reset
- `"normal"` - Normal distribution on reset
- `"constant_time_decay"` - Mass decays over time during episode
- `"action_based_decay"` - Mass decays based on action magnitude

**Key Parameters**:
- `body_name` - Which body to randomize
- `max_delta` - Maximum change for uniform mode
- `std` - Standard deviation for normal mode
- `mass_change_rate` - Decay rate for time/action-based modes
- `min_mass`, `max_mass` - Clamping bounds

**Example**:
```python
mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
    enable=True,
    randomization_modes=["normal", "constant_time_decay"],
    body_name="core",
    max_delta=0.1,
    mass_change_rate=-0.025
)
```

### 2. **Center of Mass (CoM) Randomization** (`CoMRandomizationCfg`)

**Purpose**: Randomizes the center of mass position of robot bodies to simulate payload shifts or manufacturing variations.

**Available Modes**:
- `"uniform"` - Uniform distribution on reset
- `"normal"` - Normal distribution on reset
- `"spring"` - Spring-like behavior (not implemented yet)

**Key Parameters**:
- `body_name` - Which body to randomize
- `dimension` - 2D or 3D randomization
- `max_delta` - Maximum change for uniform mode
- `std` - Standard deviation for normal mode

**Example**:
```python
com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
    enable=True,
    randomization_modes=["uniform"],
    body_name="core",
    max_delta=0.05,
    dimension=2
)
```

### 3. **Inertia Randomization** (`InertiaRandomizationCfg`)

**Purpose**: Randomizes the inertia tensor of robot bodies to simulate structural changes or manufacturing variations.

**Available Modes**:
- `"uniform"` - Uniform distribution on reset
- `"normal"` - Normal distribution on reset
- `"decay"` - Inertia decays over time

**Key Parameters**:
- `body_name` - Which body to randomize
- `max_delta` - Maximum change for uniform mode
- `std` - Standard deviation for normal mode
- `decay_rate` - Decay rate for time-based mode

**Example**:
```python
inertia_rand_cfg: InertiaRandomizationCfg = InertiaRandomizationCfg(
    enable=True,
    randomization_modes=["normal"],
    body_name="core",
    std=0.01
)
```

### 4. **Noisy Actions** (`NoisyActionsCfg`)

**Purpose**: Adds noise to robot actions to simulate sensor noise, actuator imperfections, or communication delays.

**Available Modes**:
- `"uniform"` - Uniform noise on reset, applied during actions
- `"normal"` - Normal noise on reset, applied during actions

**Key Parameters**:
- `slices` - Which action dimensions to affect
- `max_delta` - Maximum noise for uniform mode
- `std` - Standard deviation for normal mode
- `clip_actions` - Clipping bounds for actions

**Example**:
```python
noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
    enable=True,
    randomization_modes=["uniform"],
    slices=[(0, 2)],  # First 2 action dimensions
    max_delta=[0.025],
    clip_actions=[(-1, 1)]
)
```

### 5. **Actions Rescaler** (`ActionsRescalerCfg`)

**Purpose**: Scales robot actions by random factors to simulate actuator gain variations or calibration differences.

**Available Modes**:
- `"uniform"` - Uniform scaling factors on reset, applied during actions

**Key Parameters**:
- `slices` - Which action dimensions to affect
- `rescaling_ranges` - Min/max scaling factors
- `clip_actions` - Clipping bounds for actions

**Example**:
```python
actions_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
    enable=True,
    randomization_modes=["uniform"],
    slices=[(0, 2)],
    rescaling_ranges=[(0.8, 1.0)],
    clip_actions=[(-1, 1)]
)
```

### 6. **Noisy Observations** (`NoisyObservationsCfg`)

**Purpose**: Adds noise to observations to simulate sensor noise or measurement uncertainties.

**Available Modes**:
- `"uniform"` - Uniform noise on reset, applied during observations
- `"normal"` - Normal noise on reset, applied during observations

**Key Parameters**:
- `slices` - Which observation dimensions to affect
- `max_delta` - Maximum noise for uniform mode
- `std` - Standard deviation for normal mode
- `normalize` - Whether to normalize after adding noise

**Example**:
```python
noisy_observations_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
    enable=True,
    randomization_modes=["normal"],
    slices=[(0, 3)],  # First 3 observation dimensions
    std=[0.01],
    normalize=[True]
)
```

### 7. **Wrench Randomization** (`WrenchRandomizationCfg`)

**Purpose**: Applies random forces and torques to robot bodies to simulate external disturbances, wind, or contact forces.

**Available Modes**:
- `"kick_uniform"` - Sporadic kicks with uniform distribution
- `"kick_normal"` - Sporadic kicks with normal distribution
- `"constant_uniform"` - Constant wrenches with uniform distribution
- `"constant_normal"` - Constant wrenches with normal distribution
- `"constant_sinusoidal"` - Sinusoidal varying wrenches

**Key Parameters**:
- `body_name` - Which body to apply wrenches to
- `uniform_force`, `uniform_torque` - Force/torque ranges for uniform mode
- `normal_force`, `normal_torque` - Force/torque parameters for normal mode
- `kick_force_multiplier`, `kick_torque_multiplier` - Scaling for kicks
- `push_interval` - How often kicks occur
- `use_sinusoidal_pattern` - Whether to use sinusoidal pattern for constant wrenches
- `sine_wave_pattern` - Frequency range for sinusoidal mode

**Example**:
```python
# Sporadic kicks
wrench_rand_cfg: WrenchRandomizationCfg = WrenchRandomizationCfg(
    enable=True,
    randomization_modes=["kick_uniform"],
    body_name="core",
    uniform_force=(0, 0.25),
    uniform_torque=(0, 0.05),
    push_interval=5
)

# Sinusoidal varying wrenches
wrench_rand_cfg: WrenchRandomizationCfg = WrenchRandomizationCfg(
    enable=True,
    randomization_modes=["constant_sinusoidal"],
    body_name="core",
    uniform_force=(0, 0.25),
    uniform_torque=(0, 0.05),
    use_sinusoidal_pattern=True,
    sine_wave_pattern=(0.5, 2.0)  # Frequency range in Hz
)
```

## When Randomization is Applied

### **Reset-based Randomization** (applied when environment resets):
- Mass (uniform/normal modes)
- CoM (uniform/normal modes)
- Inertia (uniform/normal modes)
- Noisy Actions (uniform/normal modes)
- Actions Rescaler (uniform mode)
- Noisy Observations (uniform/normal modes)
- Wrench (kick modes, constant modes)

### **Step-based Randomization** (applied every simulation step):
- Mass (decay modes)
- Inertia (decay mode)
- Noisy Actions (noise application)
- Actions Rescaler (scaling application)
- Noisy Observations (noise application)
- Wrench (constant application, kick timing)

## Complete Configuration Example

Here's how you might configure multiple randomizations for a robust training setup:

```python
@configclass
class RobotCfg:
    # Mass randomization - changes on reset and decays over time
    mass_rand_cfg: MassRandomizationCfg = MassRandomizationCfg(
        enable=True,
        randomization_modes=["normal", "constant_time_decay"],
        body_name="core",
        max_delta=0.1,
        mass_change_rate=-0.025
    )

    # CoM randomization - changes on reset only
    com_rand_cfg: CoMRandomizationCfg = CoMRandomizationCfg(
        enable=True,
        randomization_modes=["uniform"],
        body_name="core",
        max_delta=0.05
    )

    # Action noise - adds noise to actions
    noisy_actions_cfg: NoisyActionsCfg = NoisyActionsCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 2)],  # First 2 action dimensions
        max_delta=[0.025],
        clip_actions=[(-1, 1)]
    )

    # Action scaling - varies actuator gains
    actions_rescaler_cfg: ActionsRescalerCfg = ActionsRescalerCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 2)],
        rescaling_ranges=[(0.8, 1.0)],
        clip_actions=[(-1, 1)]
    )

    # External wrenches - sporadic kicks
    wrench_rand_cfg: WrenchRandomizationCfg = WrenchRandomizationCfg(
        enable=True,
        randomization_modes=["kick_uniform"],
        body_name="core",
        uniform_force=(0, 0.25),
        uniform_torque=(0, 0.05),
        push_interval=5
    )
```

## Legacy Randomization Functions

The framework also supports legacy randomization functions from `isaaclab.envs.mdp.events`:

- `randomize_rigid_body_mass()` - Mass randomization
- `randomize_rigid_body_material()` - Material properties (friction, restitution)
- `randomize_actuator_gains()` - Joint stiffness and damping
- `randomize_joint_parameters()` - Joint friction, armature, limits
- `randomize_physics_scene_gravity()` - Global gravity
- `randomize_fixed_tendon_parameters()` - Tendon properties

## Best Practices

1. **Start Simple**: Begin with basic randomizations (mass, CoM) before adding complex ones
2. **Monitor Performance**: Track how randomization affects training stability and convergence
3. **Gradual Increase**: Gradually increase randomization intensity during training
4. **Test Robustness**: Validate policies in environments with different randomization settings
5. **Balance Realism**: Ensure randomization ranges reflect realistic variations

## Architecture

The randomization system is built on the `RandomizationCore` class which provides:
- Automatic mode detection and function mapping
- Timing control (reset vs. step-based)
- Environment-specific application
- Data logging and monitoring capabilities

Each randomization type inherits from `RandomizationCore` and implements specific behavior for its domain.
