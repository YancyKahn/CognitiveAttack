import os
import pandas as pd
import json

from typing import List, Dict

import torch
from cfgs.base_config import JailRLConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
import wandb
import numpy as np
from torch.utils.data import DataLoader

class CustomPPOTrainer:
    def __init__(self, config: JailRLConfig):
        self.config = config
        self.current_step = 0

        # Helper function to load model
        def load_model(model_path: str):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.config.device_map,
                trust_remote_code=True,
                # load_in_4bit=True
                quantization_config=bnb_config
            )
            

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        self.base_model = load_model(config.model_path)
        self.ref_model = load_model(config.model_path)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

                
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.base_model.config.pad_token_id = self.tokenizer.eos_token_id


        # LoRA configuration and model
        self.lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=config.lora_task_type,
        )
        
        lora_model = get_peft_model(self.base_model, self.lora_config)
        lora_model.print_trainable_parameters()
        
        self.red_model = AutoModelForCausalLMWithValueHead.from_pretrained(lora_model)
        
        # accelerator
        self.accelerator = Accelerator()
        self.red_model = self.accelerator.prepare(self.red_model)
        self.ref_model = self.accelerator.prepare(self.ref_model)
        
        # PPO configuration
        self.ppo_config = PPOConfig(
            learning_rate=config.rl_learning_rate,
            batch_size=config.rl_batch_size,
            mini_batch_size=config.rl_mini_batch_size,
            optimize_cuda_cache=config.rl_optimize_cuda_cache,
            adap_kl_ctrl=config.rl_adap_kl_ctrl,
            init_kl_coef=config.rl_init_kl_coef,
            kl_penalty=config.rl_kl_penalty,
            vf_coef=config.rl_vf_coef,
            cliprange=config.rl_cliprange,
            cliprange_value=config.rl_cliprange_value,
            early_stopping=config.rl_early_stopping,
            seed=config.seed,
            log_with=config.rl_log_with,
            tracker_kwargs={
                "wandb": {
                    "entity": config.wandb_project_name,
                    "name": config.wandb_run_name,
                }
            }
        )

        self.ppo_trainer = PPOTrainer(
            self.ppo_config,
            self.red_model,
            self.ref_model,
            self.tokenizer
        )

        self.memory = list()
        self.losses = list()
        self.rewards = list()
        self.logs = list()
        
        if os.path.exists(os.path.join(self.config.log_dir, "memory.jsonl")):
            # load memory
            self.load_memory()

    def save_lora_model(self):
        model_save_path = os.path.join(self.config.output_dir, f"lora_model_step_{self.current_step}")
        
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        self.red_model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"LoRA model saved to {model_save_path}")

    
    def load_lora_model(self, adapter_model_path: str=None):
        if adapter_model_path is None:
            adapter_model_path = self.config.adapter_model_path
        
        self.red_model = PeftModel.from_pretrained(self.config.model_name, 
                                                   adapter_model_path=adapter_model_path,
                                                    device_map=self.config.device_map)

    def train_step(self, batch: Dict[str, List]):
        self.current_step += 1
        # Better: use tokenizer.batch_encode_plus
        query_tensor = [self.tokenizer(p, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0) for p in batch["prompts"]]
        
        response_tensor = [self.tokenizer(r, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0) for r in batch["responses"]]
        
        rewards_tensor = [torch.tensor(r) for r in batch["rewards"]]
        
        # Perform PPO step using the prepared models and tensors
        stats = self.ppo_trainer.step(query_tensor, response_tensor, rewards_tensor)
        
        self.ppo_trainer.log_stats(stats, 
                                batch={"query": batch["prompts"], "response": batch["responses"]},
                                rewards=batch["rewards"])
        
        print(f"Step {self.current_step} | Policy Loss: {stats['ppo/loss/total']} | Reward Mean: {stats['ppo/returns/mean']}")
        
        self.losses.append(stats['ppo/loss/total'])
        self.rewards.append(stats['ppo/returns/mean'])
        
        self.save_train_logs(stats)
        
        if (self.current_step + 1) % self.config.rl_save_freq == 0:
            self.plot_metrics()
            self.save_lora_model()

    
    def save_train_logs(self, stats):
        def convert_ndarray(o):
            if isinstance(o, np.ndarray):
                return o.tolist() 
            elif isinstance(o, dict):  
                return {k: convert_ndarray(v) for k, v in o.items()}
            return o  
        
        # Convert stats (or any numpy arrays inside stats) to a format that can be logged
        stats_converted = convert_ndarray(stats)

        # Create a wandb Table to log the stats (assuming stats is a dictionary of lists or a dictionary of dicts)
        # Ensure that stats is a list of dictionaries for wandb.Table
        if isinstance(stats_converted, dict):
            # If stats is a dictionary of values (like a dictionary of lists), we need to structure it into rows
            # For example, each key-value pair in the dictionary becomes a column in the table
            # We'll transpose the dictionary to a list of rows
            columns = list(stats_converted.keys())
            row = [stats_converted[key] for key in columns]
            table = wandb.Table(columns=columns, data=[row])
        else:
            data = [{"loss": self.losses, "reward": self.rewards}]
            table = wandb.Table(columns=["loss", "reward"], data=data)
            
        wandb.log({"train_logs": table})
    
    def plot_metrics(self):
        from matplotlib import pyplot as plt
        # Plot the loss and rewards
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(list(range(len(self.losses))), list(self.losses), label="PPO Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("PPO Loss Over Time")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(list(range(len(self.rewards))), list(self.rewards), label="Reward Mean")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Mean Reward Over Time")
        plt.legend()

        wandb.log({"metrics_plot": wandb.Image(plt)})
        plt.close()

    def load_memory(self):
        if not os.path.exists(os.path.join(self.config.log_dir, "memory.jsonl")):
            return 
        with open(os.path.join(self.config.log_dir, "memory.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.memory.append(json.loads(line))
    
    def save_memory(self):
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        
        batch_save_memory = self.memory[-self.config.rl_batch_size:]
        with open(os.path.join(self.config.log_dir, "memory.jsonl"), "a+", encoding="utf-8") as f:
            for sample in batch_save_memory:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def batch_from_memory(self):
        batch_size = min(self.config.rl_batch_size, len(self.memory))
        batch = self.memory[-batch_size:]
        
        return {
            "prompts": [sample["prompt"] for sample in batch],
            "responses": [sample["response"] for sample in batch],
            "rewards": [sample["rewards"] for sample in batch]
        }

    def train_offline(self):
        if len(self.memory) < self.config.rl_batch_size:
            return
        
        dataloader = DataLoader(
                self.memory,
                batch_size=self.config.rl_batch_size,
                shuffle=False)
        
        for epoch in range(self.config.rl_num_epochs):
            for batch in dataloader:
                sample_batch = {
                    "prompts": [sample for sample in batch["prompt"]],
                    "responses": [sample for sample in batch["response"]],
                    "rewards": [sample for sample in batch["rewards"]]
                }
                self.train_step(sample_batch)