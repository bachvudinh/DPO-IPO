import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import gc
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, AutoPeftModelForCausalLM


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save_lora_params()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    bnb_config = BitsAndBytesConfig( 
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    peft_config = LoraConfig(
        r =32,
        lora_alpha = 16,
        target_modules = [
            'Wqkv',
            'out_proj',
        ],
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )
    ipo_peft_config = LoraConfig(
        r =32,
        lora_alpha = 16,
        target_modules = [
            'Wqkv',
            'out_proj',
        ],
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), trust_remote_code=True ,low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)
    lora_policy = get_peft_model(policy, peft_config)
    del policy
    gc.collect()
    # torch.cuda.empty_cache()
    lora_policy.print_trainable_parameters()
    # lora_policy.config.pretraining_tp = 1

    if config.loss.name in {'sft'}:
        lora_reference = None

    # if config.model.archive is not None:
    #     state_dict = torch.load(config.model.archive, map_location='cpu')
    #     step, metrics = state_dict['step_idx'], state_dict['metrics']
    #     print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
    #     policy.load_state_dict(state_dict['state'])
    #     if config.loss.name in {'dpo', 'ipo'}:
    #         lora_reference_model.load_state_dict(state_dict['state'])
    #     print('loaded pre-trained weights')
    if config.model.archive is not None:
        del lora_policy
        gc.collect()
        torch.cuda.empty_cache()
        output_dir = config.model.archive
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, torch_dtype=policy_dtype, **model_kwargs)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        print("Final merged checkpoint saved to %s", output_merged_dir)
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            output_merged_dir, cache_dir=get_local_dir(config.local_dirs), trust_remote_code=True ,low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        disable_dropout(policy)
        lora_policy = get_peft_model(policy, ipo_peft_config)
        reference_dtype = getattr(torch, config.model.reference_dtype)
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            output_merged_dir, cache_dir=get_local_dir(config.local_dirs), trust_remote_code=True ,low_cpu_mem_usage=True, torch_dtype=reference_dtype, **model_kwargs)
        disable_dropout(reference)
        lora_reference = get_peft_model(reference, ipo_peft_config)
        # lora_config_poli = PeftConfig.from_pretrained(config.model.archive)
        # pretrained_policy = transformers.AutoModelForCausalLM.from_pretrained(
        #     lora_config_poli.base_model_name_or_path, cache_dir=get_local_dir(config.local_dirs), trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        # disable_dropout(pretrained_policy)
        # pretrained_lora_policy = PeftModel.from_pretrained(pretrained_policy, config.model.archive)
        # lora_policy = get_peft_model(pretrained_lora_policy, ipo_peft_config)
        # if config.loss.name in {'dpo', 'ipo'}:
        #     lora_config_ref = PeftConfig.from_pretrained(config.model.archive)
        #     pretrained_reference = transformers.AutoModelForCausalLM.from_pretrained(
        #         lora_config_ref.base_model_name_or_path, cache_dir=get_local_dir(config.local_dirs), trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        #     pretrained_lora_reference = PeftModel.from_pretrained(pretrained_reference, config.model.archive)
        #     lora_reference = get_peft_model(pretrained_lora_reference, ipo_peft_config)
        print('Succesfully loaded pre-trained weights from {}'.format(output_merged_dir))

    
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, lora_policy, lora_reference), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, lora_policy, lora_reference)


if __name__ == '__main__':
    main()