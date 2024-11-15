from omni.isaac.lab.assets import ArticulationData, Articulation
from omni.isaac.lab_tasks.rans import FloatingPlatformRobotCfg
from .robot_core import RobotCore

import torch


class FloatingPlatformRobot(RobotCore):

    def __init__(
            self, 
            robot_cfg: FloatingPlatformRobotCfg, 
            robot_uid: int = 0, 
            num_envs: int = 1, 
            device: str = "cuda",
    ):
        super(FloatingPlatformRobot, self).__init__(robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        # Available for use robot_cfg.is_reaction_wheel,robot_cfg.split_thrust,robot_cfg.rew_reaction_wheel_scale
        self._dim_robot_obs = 2
        self._dim_robot_act = self._robot_cfg.num_thrusters if not self._robot_cfg.is_reaction_wheel else self._robot_cfg.num_thrusters + 1
        self._dim_gen_act = 0

        # Buffers
        self.initialize_buffers()

    def initialize_buffers(self, env_ids=None):
        super().initialize_buffers(env_ids)
        self._actions = torch.zeros((self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32)
        self._previous_actions = torch.zeros((self._num_envs, self._dim_robot_act), device=self._device, dtype=torch.float32)
        self._thrust_action = torch.zeros((self._num_envs, self._robot_cfg.num_thrusters), device=self._device, dtype=torch.float32)
        if self._robot_cfg.is_reaction_wheel:
            self._reaction_wheel_action = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)


    def run_setup(self, robot: Articulation):
        super().run_setup(robot)
        self._thrusters_dof_idx, _ = self._robot.find_bodies(self._robot_cfg.thrusters_dof_name)
        
        if self._robot_cfg.is_reaction_wheel:
            self._reaction_wheel_dof_idx, _ = self._robot.find_joints(self._robot_cfg.reaction_wheel_dof_name)

    def create_logs(self):
        super().create_logs()
        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )
        self._logs["state"]["thrusters"] = torch_zeros()
        self._logs["state"]["reaction_wheel"] = torch_zeros()
        self._logs["state"]["action_rate"] = torch_zeros()
        self._logs["state"]["joint_acceleration"] = torch_zeros()
        self._logs["reward"]["action_rate"] = torch_zeros()
        self._logs["reward"]["joint_acceleration"] = torch_zeros()

    def get_observations(self) -> torch.Tensor:
        # print robot_data positions to validate in only moves on the x-y plane
        print(f"Robot data positions: {self.body_pos_w}")

        return self._actions

    def compute_rewards(self):
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)

        # Log data
        self._logs["state"]["action_rate"] = action_rate
        self._logs["state"]["joint_acceleration"] = joint_accelerations
        self._logs["reward"]["action_rate"] = action_rate
        self._logs["reward"]["joint_acceleration"] = joint_accelerations
        return (
            action_rate * self._robot_cfg.rew_action_rate_scale
            + joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        )

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        task_done = torch.zeros(self._num_envs, dtype=torch.int32, device=self._device)
        return task_failed, task_done

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ):
        super().reset(env_ids, gen_actions, env_seeds)
        self._previous_actions[env_ids] = 0

    def reset_logs(self, env_ids: torch.Tensor) -> None:
        # Reset logs
        self._logs["state"]["thrusters"][env_ids] = 0
        self._logs["state"]["reaction_wheel"][env_ids] = 0
        self._logs["state"]["action_rate"][env_ids] = 0
        self._logs["state"]["joint_acceleration"][env_ids] = 0
        self._logs["reward"]["action_rate"][env_ids] = 0
        self._logs["reward"]["joint_acceleration"][env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor):
        thrust_reset = torch.zeros_like(self._thrust_action)
        self._robot.set_external_force_and_torque(thrust_reset, thrust_reset, body_ids=self._thrusters_dof_idx, env_ids=env_ids)
        if self._robot_cfg.is_reaction_wheel:
            rw_reset = torch.zeros_like(self._reaction_wheel_action)
            self._robot.set_joint_velocity_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
            self._robot.set_joint_effort_target(rw_reset, joint_ids=self._reaction_wheel_dof_idx, env_ids=env_ids)
        
    def process_actions(self, actions: torch.Tensor):
        # Expand to match num_envs x action_dim
        if actions.ndim == 1:  # Single environment action
            actions = actions.unsqueeze(0).expand(self._num_envs, -1)

        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        # Calculate the number of active thrusters (those with a value of 1)
        n_active_thrusters = torch.sum(actions[:, :8], dim=1, keepdim=True)  # Count 1s for active thrusters
        # Determine thrust scaling factor
        if self._robot_cfg.split_thrust:
            # Calculate thrust scale as max thrust divided by the number of active thrusters
            thrust_scale = torch.where(n_active_thrusters > 0,
                                    self._robot_cfg.max_thrust / n_active_thrusters,
                                    torch.tensor(0.0, device=actions.device))
        else:
            thrust_scale = self._robot_cfg.max_thrust

        # Apply thrust to thrusters, based on whether reaction wheel is present
        self._thrust_action = actions[:, :self._robot_cfg.num_thrusters].float() * thrust_scale
        # Expand thrust action dimensions to match 3D force and torque (x, y, z) for all envs
        self._thrust_action = self._thrust_action.unsqueeze(2).expand(-1, -1, 3)

        if self._robot_cfg.is_reaction_wheel:
            # Separate continuous control for reaction wheel
            self._reaction_wheel_action = actions[:, self._robot_cfg.num_thrusters:] * self._robot_cfg.reaction_wheel_scale
            self._reaction_wheel_action = self._reaction_wheel_action.unsqueeze(2).expand(-1, -1, 3)



        # Log data for monitoring
        self._logs["state"]["thrusters"] = self._thrust_action[:, 0]
        if self._robot_cfg.is_reaction_wheel:
            self._logs["state"]["reaction_wheel"] = self._reaction_wheel_action[:, 0]


    def compute_physics(self):
        pass  # Model motor + ackermann steering here

    def apply_actions(self, articulations: Articulation):
        articulations.set_external_force_and_torque(self._thrust_action, torch.zeros_like(self._thrust_action), body_ids=self._thrusters_dof_idx)
        if self._robot_cfg.is_reaction_wheel:
            articulations.set_joint_effort_target(self._reaction_wheel_action, joint_ids=self._reaction_wheel_dof_idx)
