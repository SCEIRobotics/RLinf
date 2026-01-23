import torch
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, fields, is_dataclass

def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.flower.flower_action_model import FlowerRLConfig, FlowerForRLActionPrediction
    model_path = cfg.get("model_path", None)

    def filter_dataclass_kwargs(cls, kwargs):
        """通用函数：过滤参数，包含当前类 + 所有父类的 dataclass 字段"""
        all_fields = set()
        for base_cls in cls.__mro__:
            if is_dataclass(base_cls) and base_cls is not object:
                all_fields.update({f.name for f in fields(base_cls)})
        return {k: v for k, v in kwargs.items() if k in all_fields}
    # filtered_cfg = filter_dataclass_kwargs(FlowerRLConfig, cfg)
    config = FlowerRLConfig(**cfg.flower)
    model = FlowerForRLActionPrediction(config)
    if model_path is not None:
        model_dict = torch.load(model_path, map_location='cpu')
        print(f"Using checkpoint for init flower model (from get_model): {model_path}")
        model.load_state_dict(model_dict)
    model.to(torch_dtype)
    if cfg.flower.train_expert_only:
        print("Freezing Flower VLM parameters")
        model.freeze_vlm()

    return model