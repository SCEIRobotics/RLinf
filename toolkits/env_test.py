import os
import hydra
# from rlinf.config import validate_cfg
from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.envs import action_utils
import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ['EMBODIED_PATH'] = '/mnt/data/xingchen/github/RLinf/examples/embodiment/config'

def env_test(cfg):
    env = ManiskillEnv(cfg.env.eval, seed_offset=0, total_num_processes=1)
    extracted_obs, _, _, _, infos = env.step()
    device = extracted_obs['images'].device
    # print(extracted_obs)
    # obs = extracted_obs['images'][0].cpu().numpy()
    # obs = cv2.cvtColor(obs.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # cv2.imwrite('logs/temp/obs.jpg', obs)

    import ipdb; ipdb.set_trace()

    max_step = 50
    for eval_step in tqdm(range(max_step)):
        action = torch.randn(cfg.env.eval.num_envs, cfg.actor.model.action_dim).to(device)
        extracted_obs, step_reward, terminations, truncations, infos = env.step(action)
    env.flush_video()

def model_test(cfg):
    # env
    env = ManiskillEnv(cfg.env.eval, seed_offset=0, total_num_processes=1)
    extracted_obs, _, _, _, infos = env.step()
    device = extracted_obs['images'].device

    # model
    model = get_model(cfg.rollout.model_dir, cfg.actor.model)
    model_config, input_processor = get_vla_model_config_and_processor(cfg.actor)
    model.setup_config_and_processor(model_config, cfg, input_processor)
    model.to(device)
    model.eval()

    import ipdb; ipdb.set_trace()

    # inference
    max_step = cfg.algorithm.n_eval_chunk_steps
    kwargs = dict(cfg.algorithm.sampling_params)
    kwargs["do_sample"] = True
    kwargs["use_cache"] = True
    kwargs["temperature"] = kwargs["temperature_eval"]
    for _ in tqdm(range(max_step)):
        extracted_obs["states"] = None
        extracted_obs["wrist_images"] = None
        actions, result = model.predict_action_batch(env_obs=extracted_obs, **kwargs)
        chunk_actions = action_utils.prepare_actions_for_maniskill(
            actions,
            num_action_chunks=cfg.actor.model.num_action_chunks,
            action_dim=cfg.actor.model.action_dim,
            policy=cfg.actor.model.get("policy_setup", None),
            action_scale=1.0)
        extracted_obs, step_reward, terminations, truncations, infos = env.chunk_step(chunk_actions)
    env.flush_video()

@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft_quickstart_test"
)
def main(cfg) -> None:
    # cfg = validate_cfg(cfg)
    cfg.env.eval.init_params.control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    cfg.env.eval.num_envs = 2
    cfg.env.eval.video_cfg.video_base_dir = 'logs/temp/'

    # env_test(cfg)
    model_test(cfg)

if __name__ == "__main__":
    main()