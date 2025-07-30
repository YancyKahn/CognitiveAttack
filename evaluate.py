import pandas as pd
import json
import argparse
import time
from judges import GPTJudge, LlamaGuardJudge
import config
from openai import OpenAI
import random
import os
from cfgs.base_config import JailRLConfig, RedteamConfig, JudgeConfig
from train import PPOTrainer
import torch
from models.redteam import RedTeamModel
from utils.data_tools import PromptLoader
from utils.data_tools import custom_collate_fn_rl
from torch.utils.data import DataLoader
from utils.reward_calculator import RewardCalculator

from utils.data_tools import safety_check

from pathlib import Path

BASE_TASK_ID = "sft-attack/cognitive-bias-attack"

dataset_map = {
    "advbench": {
        "input": r"YOUR_PATH_TO/data/advbench/harmful_behaviors.csv"
    },
    "advbench-sft": {
        "input": r"YOUR_PATH_TO/data/SFT/multi-bias/advbench-multi-bias-input.jsonl",
        "meta": r"YOUR_PATH_TO/data/SFT/multi-bias/advbench-multi-bias-meta.jsonl"
    },
    "advbench-sft-rl": {
        "input": r"YOUR_PATH_TO/data/SFT+RL/task-advbench.jsonl",
    }
}

def read_data_sft(dataset_name):
    try:
        input_path = Path(dataset_map[dataset_name]["input"])
        meta_path = Path(dataset_map[dataset_name]["meta"])
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        print(f"Reading files:\nInput: {input_path}\nMeta: {meta_path}")
        
        input_data = pd.read_json(input_path, lines=True, encoding='utf-8')
        meta_data = pd.read_json(meta_path, lines=True, encoding='utf-8')
        
        if "custom_id" not in input_data.columns:
            raise ValueError("'custom_id' column not found in input data")
        if "custom_id" not in meta_data.columns:
            raise ValueError("'custom_id' column not found in meta data")
        
        merged_data = pd.merge(
            input_data, 
            meta_data, 
            on="custom_id",
            how="inner" 
        )
        
        print(f"Successfully merged data. Total records: {len(merged_data)}")
        
        data_list = merged_data.to_dict('records') 
        new_data = []
        for item in data_list:
            new_data.append({
                "custom_id": item["custom_id"],
                "redteam": {
                    "original_prompt": item["instruction"],
                    "enhanced_prompt": item["body"]["messages"][0]["content"]
                }
            })
            
        return new_data
    
    except Exception as e:
        print(f"Error reading or merging data: {str(e)}")
        raise  

def read_data_rl(dataset_name):
    input_data = pd.read_json(dataset_map[dataset_name]["input"], lines=True, encoding='utf-8')
    return input_data.to_dict('records')

def read_data_csv(dataset_name):
    input_data = pd.read_csv(dataset_map[dataset_name]["input"], encoding='utf-8')
    data_list = input_data["instruction"].tolist()
    
    new_data_list = []
    
    for item in data_list:
        new_data_list.append({
            "custom_id": str(random.randint(100000, 999999)), 
            "redteam": {
                "original_prompt": item,
                "enhanced_prompt": item
            }
        })

    return new_data_list
    
def load_data(dataset_name):
    if dataset_name == "advbench-sft-rl":
        return read_data_rl(dataset_name)
    elif dataset_name == "advbench-sft":
        return read_data_sft(dataset_name)
    else:
        return read_data_csv(dataset_name)
def build_redteam_biases(rl_config, prompts, trainer):
    biases_list = []
    
    for prompt in prompts:
        tokenized = trainer.policy.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'].to(rl_config.device)
        attention_mask = tokenized['attention_mask'].to(rl_config.device)
        
        with torch.no_grad():
            outputs = trainer.policy(input_ids, attention_mask)
            
        selected_indices = trainer.sample_biases(outputs['strategy_logits'])
        
        selected_biases = [trainer.bias_strategy_list[idx] for idx in selected_indices[0]]
        
        biases_list.append(selected_biases)
    
    return biases_list
        
def attack(args):
    
    for model_name in args.models:
        
        TASK_ID = BASE_TASK_ID + '/' + model_name
        

        rt_config = RedteamConfig(
            TASK_ID=TASK_ID,
            # model_name="deepseek-r1",
            task_type="evaluate-api",
        )
        redteam_model = RedTeamModel(rt_config)
        redteam_model.setup_model()
        
        judge_config = JudgeConfig(TASK_ID=TASK_ID,
                                   base_url=config.DASHSCOPE_API_BASE,
                                   api_key=config.DASHSCOPE_API_KEY,
                                   model_name="deepseek-v3")
        
        reward_calculator = RewardCalculator(judge_config)
        
    
        id = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        model_name_str = model_name.replace("/", "-")
        save_path = os.path.join(f"logs/{TASK_ID}", f"task-{args.dataset_name}-{model_name_str}-{id}.jsonl")

        data = load_data(args.dataset_name)
        
        total_samples = len(data)
        num_batches = (total_samples + args.batch_size - 1) // args.batch_size 
        
        
        print(f"Attack model: {model_name}")
        print(f"Save path: {save_path}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, total_samples)
            
            batch = data[start_idx:end_idx]
            generated_prompt = batch
            
            print("Launching Attacks ...")
            jailbreak_response = redteam_model.launch_attacks(generated_prompt, [rt_config.get_agent(model_name)])
            
            print("Judging ...")
            judge_response = reward_calculator.gptjudge_batch_score(jailbreak_response)
            
            result = jailbreak_response
            for i in range(len(jailbreak_response)):
                result[i]["judge_score"] = judge_response[i]["score"]
            
            print(f"Saving to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as f:
                for res in result:
                    f.write(json.dumps(safety_check(res), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--models", type=str, nargs="+", default=[
        "gpt-4o-mini",
        # "gpt-4o",
        # "gemini-gemini-1.5-flash",
        # "my-claude-3-haiku-20240307",
        # "siliconflow-QwQ-32B-Preview",
        # "siliconflow-QwQ-32B-Preview",
        # "dashscope-qwq-32b-preview",
        # "custom-o1-mini",
        # "deepseek-deepseek-chat",
        # "ollama-huihui_ai/skywork-o1-abliterated",
        # "ollama-qwq",
        # "dashscope-llama3.1-70b-instruct",
        # "dashscope-llama3.1-405b-instruct",
        # "cloudflare-llama-3.1-70b-instruct",
        # "dashscope-deepseek-v3",
        # "dashscope-deepseek-r1",
        # "dashscope-qwen-plus",
        # "dashscope-qwen-turbo",
        # "dashscope-qwen-max",
        # "meta-llama/Llama-3.3-70B-Instruct"
        # "qwq"
        # "deepseek-r1-distill-llama-8b"
        # "deepseek-v3-241226"
        # "meta-llama/Llama-3.3-70B-Instruct"
        # "llama2"
        # "qwen-turbo"
        ]
    )
    parser.add_argument("--dataset_name", type=str, default="advbench-sft")
    
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for evaluation")
    
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="YOUR_PATH_TO/checkpoint.pt")
    
    args = parser.parse_args()
    
    attack(args)