import os
import hydra
# from rlinf.config import validate_cfg
from rlinf.config import SupportedModel
from rlinf.envs import get_env_cls
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.envs.action_utils import prepare_actions
# from rlinf.envs import action_utils
import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ['EMBODIED_PATH'] = '/mnt/data/xingchen/github/RLinf/examples/embodiment/config'

def to_gpu(datas: dict, device) -> dict:
    for k, v in datas.items():
        if isinstance(v, torch.Tensor):
            datas[k] = v.to(device)
        else:
            datas[k] = v
    return datas

def env_test(cfg, num_envs):
    eval_env_cls = get_env_cls(cfg.env.eval.simulator_type, cfg.env.eval)
    env = eval_env_cls(cfg.env.eval, num_envs = num_envs, seed_offset=0, total_num_processes=1)
    extracted_obs, infos = env.reset()
    device = extracted_obs['images'].device
    # print(extracted_obs)
    # obs = extracted_obs['images'][0].cpu().numpy()
    # obs = cv2.cvtColor(obs.transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # cv2.imwrite('logs/temp/obs.jpg', obs)

    # import ipdb; ipdb.set_trace()

    max_step = 50
    for eval_step in tqdm(range(max_step)):
        action = torch.randn(num_envs, cfg.actor.model.action_dim).to(device)
        extracted_obs, step_reward, terminations, truncations, infos = env.step(action)
    env.flush_video()

def model_test(cfg, num_envs, model_dir, device):
    # model path update
    import ipdb; ipdb.set_trace()
    # cfg.rollout.model.model_path = model_dir
    # cfg.actor.model.model_path = model_dir
    # cfg.actor.tokenizer.tokenizer_model = model_dir

    # env
    eval_env_cls = get_env_cls(cfg.env.eval.simulator_type, cfg.env.eval)
    env = eval_env_cls(cfg.env.eval, num_envs = num_envs, seed_offset=0, total_num_processes=1)
    extracted_obs, infos = env.reset()

    # model
    model = get_model(cfg.actor.model)
    if SupportedModel(cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
        model_config, input_processor = get_vla_model_config_and_processor(cfg.actor)
        model.setup_config_and_processor(model_config, cfg, input_processor)
    if model_dir is not None:
        model_dict = torch.load(model_dir, map_location='cpu')
        model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # inference
    max_step = cfg.env.eval.max_episode_steps // cfg.actor.model.num_action_chunks
    # max_step = 10
    kwargs = dict(cfg.algorithm.sampling_params)
    kwargs["do_sample"] = True
    kwargs["use_cache"] = True
    kwargs["temperature"] = kwargs["temperature_eval"]
    for _ in tqdm(range(max_step)):
        # extracted_obs["states"] = None
        # extracted_obs["wrist_images"] = None
        model_inputs = to_gpu(extracted_obs, device)
        actions, result = model.predict_action_batch(env_obs=model_inputs, mode="train", **kwargs)
        chunk_actions = prepare_actions(
            raw_chunk_actions=actions,
            simulator_type=cfg.env.train.simulator_type,
            model_type=cfg.actor.model.model_type,
            num_action_chunks=cfg.actor.model.num_action_chunks,
            action_dim=cfg.actor.model.action_dim,
            policy=cfg.actor.model.get("policy_setup", None),)
        extracted_obs, step_reward, terminations, truncations, infos = env.chunk_step(chunk_actions)
    env.flush_video()

@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft_quickstart_test"
)
def main(cfg) -> None:
    # cfg = validate_cfg(cfg)
    if cfg.env.eval.simulator_type == 'maniskill':
        cfg.env.eval.init_params.control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    cfg.env.eval.video_cfg.video_base_dir = 'logs/temp/libero_flower'

    device = 'cuda:0'
    model_dir = '/mnt/data/xingchen/models/flower_train/avg_seq_len=0.93_valuehead.ckpt'
    model_dir = None
    # env_test(cfg, num_envs = 2)
    # model_test(cfg, num_envs = 2, model_dir = '/mnt/data/xingchen/github/RLinf/logs/20251212-08:44:22/test_openvla/checkpoints/global_step_75/actor/model')
    model_test(cfg, num_envs = 4, model_dir = model_dir, device=device)

if __name__ == "__main__":
    main()