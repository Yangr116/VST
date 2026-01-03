

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoProcessor
    from .config import PPOConfig
    from ..utils.dataset import RLHFDataset, RLHFDatasetVLM3D, collate_fn

    config_path = "/opt/tiger/rayyang/spatial_rl/examples/config_vlm3d_reasoner.yaml"
    default_config = OmegaConf.structured(PPOConfig())
    file_config = OmegaConf.load(config_path)
    default_config = OmegaConf.merge(default_config, file_config)
    ppo_config: PPOConfig = OmegaConf.to_object(default_config)
    ppo_config.deep_post_init()

    config = ppo_config.data
    model_path = "/mnt/hdfs/rayyang/hf_model/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    data_file = "/mnt/hdfs/rayyang/spatial_data/parquet/reason/rl/multi_view_reason_camcam1k_ori2k_scannetpp"
    train_dataset = RLHFDatasetVLM3D(
        data_path=data_file,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        think_prompt_in_system=config.think_prompt_in_system,
        image_special_token=config.image_special_token,
        video_special_token=config.video_special_token,
    )

    for sample in train_dataset:
        import pdb; pdb.set_trace()
