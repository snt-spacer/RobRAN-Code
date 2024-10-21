from omni.isaac.lab.assets import ArticulationData, Articulation

from omni.isaac.lab_tasks.rans import LeatherbackRobotCfg
from .robot_core import RobotCore

import torch


class LeatherbackRobot(RobotCore):
    def __init__(self, robot_cfg: LeatherbackRobotCfg, robot_uid: int = 0, num_envs: int = 1, device: str = "cuda"):
        super(LeatherbackRobot, self).__init__(robot_cfg, robot_uid=robot_uid, num_envs=num_envs, device=device)
        self._robot_cfg = robot_cfg
        self._dim_robot_obs = 2
        self._dim_robot_act = 2
        self._dim_task_act = 0

        self._actions = torch.zeros((num_envs, self._dim_robot_act), device=device, dtype=torch.float32)
        self._previous_actions = torch.zeros((num_envs, self._dim_robot_act), device=device, dtype=torch.float32)
        self._throttle_action = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)
        self._steering_action = torch.zeros((num_envs, 1), device=device, dtype=torch.float32)

    def run_setup(self, articulation: Articulation):
        self._throttle_dof_idx, _ = articulation.find_joints(self._robot_cfg.throttle_dof_name)
        self._steering_dof_idx, _ = articulation.find_joints(self._robot_cfg.steering_dof_name)

    def create_logs(self):
        super(LeatherbackRobot, self).create_logs()
        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False
        )
        self._logs["state"]["throttle"] = torch_zeros()
        self._logs["state"]["steering"] = torch_zeros()
        self._logs["state"]["action_rate"] = torch_zeros()
        self._logs["state"]["joint_acceleration"] = torch_zeros()
        self._logs["reward"]["action_rate"] = torch_zeros()
        self._logs["reward"]["joint_acceleration"] = torch_zeros()

    def get_observations(self, robot_data: ArticulationData) -> torch.Tensor:
        return self._actions

    def compute_rewards(self, robot_data: ArticulationData):
        # TODO: DT should be factored in?

        # Compute
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(robot_data.joint_acc), dim=1)

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
        self, task_actions: torch.Tensor, env_seeds: torch.Tensor, articulations: Articulation, env_ids: torch.Tensor
    ):
        self._actions[env_ids] = 0
        self._previous_actions[env_ids] = 0
        # Updates the seed
        self._seeds[env_ids] = env_seeds

        # Reset logs
        self._logs["state"]["throttle"][env_ids] = 0
        self._logs["state"]["steering"][env_ids] = 0
        self._logs["state"]["action_rate"][env_ids] = 0
        self._logs["state"]["joint_acceleration"][env_ids] = 0
        self._logs["reward"]["action_rate"][env_ids] = 0
        self._logs["reward"]["joint_acceleration"][env_ids] = 0

    def set_initial_conditions(self, env_ids: torch.Tensor, articulations: Articulation):
        throttle_reset = torch.zeros_like(self._throttle_action)
        steering_reset = torch.zeros_like(self._steering_action)
        articulations.set_joint_velocity_target(throttle_reset, joint_ids=self._throttle_dof_idx, env_ids=env_ids)
        articulations.set_joint_position_target(steering_reset, joint_ids=self._steering_dof_idx, env_ids=env_ids)

    def process_actions(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self._robot_cfg.throttle_scale
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self._robot_cfg.steering_scale

        # Log data
        self._logs["state"]["throttle"] = self._throttle_action[:, 0]
        self._logs["state"]["steering"] = self._steering_action[:, 0]

    def compute_physics(self):
        pass  # Model motor + ackermann steering here

    def apply_actions(self, articulations: Articulation):
        articulations.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        articulations.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)
