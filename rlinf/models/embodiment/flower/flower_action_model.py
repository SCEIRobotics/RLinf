import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as trans_F
from torchvision import transforms
from lerobot.policies.flower.modeling_flower import FlowerModel
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.datasets.streaming_dataset import FlowerDataCollator

from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

@dataclass
class FlowerRLConfig(FlowerConfig):
    # config for rl
    config_name: str = (
        "flower"  # pi0_libero, pi05_libero, pi0_metaworld, pi05_metaworld
    )

    robot_type: str = "genie1"
    img_size: int = 224

    # noise configs
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]

    # hyper-parameters
    action_chunk: int = 16  # action chunk
    action_env_dim: int = 7  # for environment action dim
    max_action_dim: int = 16  # for max action dim
    # num_steps: int = 10  # denoise steps

    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection

    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    def __post_init__(self):
        object.__setattr__(self, "action_chunk", self.horizon)
    
class FlowerTokenizer(FlowerDataCollator):
    def __init__(self, vlm_path: str):
        super().__init__(vlm_path)
        self.processor = None

    def __call__(self, batch, robot_type: str):
        task_batch = batch["task_descriptions"]
        robot_batch = [robot_type] * len(task_batch)

        constructed_prompts, batch_action_index = self.construct_prompts(task_batch, robot_batch)
        text_inputs = self.tokenizer(
            constructed_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        result = {}
        result['text_input_ids'] = text_inputs['input_ids']
        result['text_attention_mask'] = text_inputs.data["attention_mask"]
        result['action_index'] = batch_action_index
        return result

class FlowerForRLActionPrediction(FlowerModel, BasePolicy):
    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "Florence2ForConditionalGeneration",
            "TimestepEmbedder",
            "FreqEmbedder",
            "ActionSpaceEmbedderParameter"
        ]
    def __init__(self, config: FlowerRLConfig):
        super().__init__(config)
        self.global_step = 0

        # value head
        proj_width = 1024
        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            
            # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.max_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
        
        # tokenizer
        self.tokenizer = FlowerTokenizer(config.vlm_path)

        # other
        self.img_transform = transforms.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],  # RGB 均值
                            std=[0.26862954, 0.26130258, 0.27577711]    # RGB 标准差
                        )
    
    def predict_action_batch(
        self, env_obs, 
        mode: Literal["train", "eval"] = "train", 
        compute_values=True, 
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        
        env_obs = self.precision_processor(env_obs)
        processed_obs = self.preprocess_observations(env_obs)
        # sample actions
        outputs = self.sample_actions(
            processed_obs, mode=mode, compute_values=compute_values
        )

        # post process actions
        actions = outputs["actions"][:, :self.config.action_chunk, :self.config.action_env_dim].cpu().numpy()
        actions = self.postprocess_actions(actions)

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }
        forward_inputs.update(processed_obs)
        
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def set_global_step(self, global_step):
        self.global_step = global_step
    
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
    
    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs
    
    def preprocess_observations(self, observation, rotate = True):
        # match lerobot inputs
        img_size = self.config.img_size
        obs_out = {}
        images = torch.stack([observation['main_images'].permute(0, 3, 1, 2).contiguous(), 
                              observation['wrist_images'].permute(0, 3, 1, 2).contiguous()], dim = 1)
        obs_out['observation.images'] = images.unsqueeze(1)  # (B, n_obs_steps, num_cameras, C, H, W)
        obs_out['observation.state'] = observation['states'].unsqueeze(1)  # (B, n_obs_steps, state_dim)

        # resize images, normalize
        batch_size, n_obs_steps, cam, channels, height, width = obs_out['observation.images'].shape
        x_reshaped = obs_out['observation.images'].view(
                    batch_size*n_obs_steps*cam,
                    channels,
                    height,
                    width,
                )
        # x_reshaped = x_reshaped[:, [2, 1, 0], :, :].contiguous()  # BGR to RGB
        if rotate:
            x_reshaped = trans_F.vflip(trans_F.hflip(x_reshaped))   # rotate 180
        x_reshaped = F.interpolate(x_reshaped.type(torch.float32), size=(img_size,img_size), mode='bilinear') / 255.
        x_reshaped = self.img_transform(x_reshaped)
        obs_out['observation.images'] = x_reshaped.view(
                    batch_size,
                    n_obs_steps,
                    cam,
                    channels,
                    img_size,
                    img_size,
                )
        
        # process text
        text_inputs = self.tokenizer(observation, robot_type=self.config.robot_type)
        obs_out.update(text_inputs)

        return obs_out
    
    def postprocess_actions(self, actions):
        return actions
    
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        self.eval()
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        self.device = chains.device

        # encoder
        cond = self.encode_observations(data)

        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            cond,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }
    
    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation["observation.images"].shape[0]
        device = next(self.parameters()).device
        self.device = device
        if noise is None:
            actions_shape = (bsize, self.config.action_chunk, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Encode image features and concatenate them all together along with the state vector.
        """
        encode_observations(observation)
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        cond = self.encode_observations(observation)

        # run sampling
        samples = self.conditional_sample(bsize, device, mode=mode,cond=cond, noise=noise, compute_values=compute_values)
        # actions = super().conditional_sample(bsize, cond=cond, noise=noise)
        # samples["actions"] = actions
        # samples["actions"] = torch.clamp(samples["actions"], -1, 1)

        return samples
    
    def conditional_sample(
        self,
        batch_size: int,
        device,
        mode="train",
        cond: Tensor | None = None,
        noise: Tensor | None = None,
        compute_values=True,
    ) -> Tensor:
        num_steps = self.num_inference_steps
        # Sample prior.
        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Integration
        dt = 1.0 / num_steps
        dt_tensor = torch.tensor([dt] * batch_size, device=device) # (batch_size,)
        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)

        for idx in range(num_steps):
            # Predict velocity field
            t_val = timesteps[idx]
            t_tensor = torch.full((batch_size,), t_val, device=device)  # (batch_size,)
            v_t, v_t_emb = self.dit_forward(x_t, t_tensor, cond)

            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                v_t,
                v_t_emb,
                t_tensor,
                dt_tensor,
                sample_mode,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        
        # process results
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }
    
    def get_log_prob_value(
            self, 
            cond,
            chains,
            denoise_inds,
            compute_values):
        
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        bsize = chains.shape[0]
        device = chains.device
        max_step = self.config.num_inference_steps
        # get init log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_inference_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0, : self.config.action_chunk, : self.config.action_env_dim],
                torch.zeros_like(chains[:, 0, : self.config.action_chunk, : self.config.action_env_dim]),
                torch.ones_like(chains[:, 0, : self.config.action_chunk, : self.config.action_env_dim]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0, : self.config.action_chunk, : self.config.action_env_dim]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        
        # pre compute
        dt = 1.0 / self.config.num_inference_steps
        dt_tensor = torch.tensor([dt] * bsize, device=device) # (batch_size,)
        timesteps = torch.linspace(1, 1 / max_step, max_step, device=device)

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]    # denoise_ind is descend
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]

            # Predict velocity field
            t_tensor = timesteps[denoise_ind]
            v_t, v_t_emb = self.dit_forward(chains_pre, t_tensor, cond)

            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                v_t,
                v_t_emb,
                t_tensor,
                dt_tensor,
                "train",
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next[:, : self.config.action_chunk, : self.config.action_env_dim], 
                                              x_t_mean[:, : self.config.action_chunk, : self.config.action_env_dim],
                                              x_t_std[:, : self.config.action_chunk, : self.config.action_env_dim])  # just compute env action dim
            entropy = self.gaussian_entropy(x_t_std[:, : self.config.action_chunk, : self.config.action_env_dim])
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy
    
    def sample_mean_var_val(self, 
                            x_t,
                            v_t,
                            v_t_emb, 
                            t_tensor,
                            dt_tensor,
                            mode,
                            compute_values):
        device = x_t.device

        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)

        # value prediction
        if (
            self.config.add_value_head
            and compute_values
        ):
            suffix_out = v_t_emb
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((v_t.shape[0]), device=device)

        # ode sde mix sampling
        delta = dt_tensor[:, None, None].expand_as(x_t)
        t_input = t_tensor[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigma_i = (
                    noise_level
                    * torch.sqrt(
                        t_input
                        / (1 - torch.where(t_input == 1, t_input-delta, t_input))
                    )
                )
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(v_t)  # TODO: need to change
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
            
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.vlm.eval()
            # self.frequency_embedder.eval()
            # self.action_space_embedder.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
            # for params in self.frequency_embedder.parameters():
            #     params.requires_grad = False
            # for params in self.action_space_embedder.parameters():
            #     params.requires_grad = False
                
if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    # model
    config = FlowerRLConfig()
    config.n_obs_steps = 1
    config.vlm_path = "/mnt/data/xingchen/models/pretrained/Florence-2-large"
    config.add_value_head = True
    config.max_action_dim = 7
    config.robot_type = 'lift2'
    config.use_proprio = False
    model = FlowerForRLActionPrediction(config)
    model_dict = torch.load("/mnt/data/xingchen/models/flower_train/avg_seq_len=0.93_valuehead.ckpt", map_location='cpu', weights_only = False)
    model.load_state_dict(model_dict)
    print(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # data
    imgs = torch.randn(2, 3, 112, 112).to(device)
    states = torch.randn(2, 7).to(device)
    prompts = ['put this cup on the table', 'move the block to the left']
    obs = {'main_images': imgs, 'wrist_images': imgs.clone(), 'states': states, 'task_descriptions': prompts}

    # inference
    actions, results = model.predict_action_batch(obs)
    print(actions.shape)

    # forward
    process_input = {}
    for key, value in results['forward_inputs'].items():
        process_input[key] = value.to(device)
    results.update(process_input)
    outs = model(results)
    ratio = outs['logprobs'] / results['prev_logprobs']
    print(outs['logprobs'].shape, ratio.shape)