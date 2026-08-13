import glob
import os

import numpy as np

from ._base_task import Base_Task
from .utils import ArmTag, create_actor, rand_pose


OBJECT_LIST = (
    "047_mouse",
    "048_stapler",
    "050_bell",
    "057_toycar",
    "073_rubikscube",
    "075_bread",
    "077_phone",
    "081_playingcards",
    "086_woodenblock",
    "112_tea-box",
    "113_coffee-box",
    "107_soap",
)


def _available_model_ids(model_name):
    pattern = os.path.join("assets", "objects", model_name, "model_data*.json")
    model_ids = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        try:
            model_ids.append(int(name.removeprefix("model_data").removesuffix(".json")))
        except ValueError:
            continue
    return sorted(model_ids)


def _pose_record(pose):
    return {"position": pose.p.tolist(), "quaternion": pose.q.tolist()}


class PlaceA2BSharedTask(Base_Task):
    """Left/right language goals over an identical, seed-addressed initial scene."""

    goal_direction = None
    goal_offset_x = 0.10

    def setup_demo(self, **kwargs):
        if self.goal_direction not in {"left", "right"}:
            raise ValueError(f"Unsupported A2B goal direction: {self.goal_direction}")
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Both arms must be able to grasp the same source scene.  Separating source
        # and target along y avoids turning one language branch into a cross-body
        # reach while preserving identical initial visual evidence.
        for _ in range(100):
            source_pose = rand_pose(
                xlim=[-0.04, 0.04],
                ylim=[-0.22, -0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            target_pose = rand_pose(
                xlim=[-0.02, 0.02],
                ylim=[-0.04, 0.03],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            source_xy = source_pose.p[:2]
            target_xy = target_pose.p[:2]
            left_goal = target_xy + np.array([-self.goal_offset_x, 0.0])
            right_goal = target_xy + np.array([self.goal_offset_x, 0.0])
            if (
                abs(source_xy[1] - target_xy[1]) >= 0.12
                and np.linalg.norm(source_xy - target_xy) >= 0.12
                and np.linalg.norm(source_xy - left_goal) >= 0.1
                and np.linalg.norm(source_xy - right_goal) >= 0.1
            ):
                break
        else:
            raise RuntimeError("Unable to sample a shared A2B scene in 100 attempts")

        self.selected_modelname_A = str(np.random.choice(OBJECT_LIST))
        source_model_ids = _available_model_ids(self.selected_modelname_A)
        if not source_model_ids:
            raise ValueError(f"No model_data files found for {self.selected_modelname_A}")
        self.selected_model_id_A = int(np.random.choice(source_model_ids))

        target_names = [name for name in OBJECT_LIST if name != self.selected_modelname_A]
        self.selected_modelname_B = str(np.random.choice(target_names))
        target_model_ids = _available_model_ids(self.selected_modelname_B)
        if not target_model_ids:
            raise ValueError(f"No model_data files found for {self.selected_modelname_B}")
        self.selected_model_id_B = int(np.random.choice(target_model_ids))

        self.object = create_actor(
            scene=self,
            pose=source_pose,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )
        self.target_object = create_actor(
            scene=self,
            pose=target_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )
        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

        self.a2b_scene_spec = {
            "schema_version": "g4-a2b-shared-scene-v1",
            "goal_offset_x": self.goal_offset_x,
            "expert_arm_assignment": "goal-conditioned",
            "source": {
                "model_name": self.selected_modelname_A,
                "model_id": self.selected_model_id_A,
                "pose": _pose_record(source_pose),
            },
            "target": {
                "model_name": self.selected_modelname_B,
                "model_id": self.selected_model_id_B,
                "pose": _pose_record(target_pose),
            },
        }

    def play_once(self):
        arm_tag = ArmTag(self.goal_direction)
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.target_object.get_pose().p.tolist()
        target_delta_x = -self.goal_offset_x if self.goal_direction == "left" else self.goal_offset_x
        target_pose[0] += target_delta_x
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))

        self.info["pair_id"] = (
            f"g4-a2b-v1:{self.task_config}:seed-{self.info['domain_randomization']['seed']}"
        )
        self.info["language_goal"] = self.goal_direction
        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        target_pose = self.target_object.get_pose().p
        distance = np.linalg.norm(object_pose[:2] - target_pose[:2])
        side_ok = object_pose[0] < target_pose[0] if self.goal_direction == "left" else object_pose[0] > target_pose[0]
        return bool(
            0.08 < distance < 0.2
            and side_ok
            and abs(object_pose[1] - target_pose[1]) < 0.05
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
