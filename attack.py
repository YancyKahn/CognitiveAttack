import json
import argparse
import tqdm
import time
import os
import logging
from cfgs.base_config import RedteamConfig, JudgeConfig
from models.redteam import RedTeamModel
from utils.data_tools import PromptLoader
from torch.utils.data import DataLoader
from utils.reward_calculator import RewardCalculator
from utils.data_tools import safety_check
import config

BASE_TASK_ID = "attack-results"

def setup_logging(log_dir: str):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/attack.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def generate_save_path(args, model_name, phase, timestamp):
    """Generate the save path for attack results."""
    model_name_str = model_name.replace("/", "-")
    return os.path.join(f"logs/redteam/{BASE_TASK_ID}/{model_name}", 
                        f"task-{phase}-{args.dataset_name}-{model_name_str}-{timestamp}.jsonl")

def load_redteam_model(args, rt_config):
    """Load the Red Team model."""
    redteam_model = RedTeamModel(rt_config)
    redteam_model.load_redteam_model(adapter_model_path=args.adapter_model_path)
    return redteam_model

def handle_full_phase(dataloader, redteam_model, rt_config, reward_calculator, model_name, save_path):
    """Handle the logic for the full attack phase."""
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        logging.info("Generating prompt for Red Team...")
        generated_prompt = redteam_model.generate(batch['instruction'])
        
        logging.info("Launching attacks...")
        jailbreak_response = redteam_model.launch_attacks(generated_prompt, [rt_config.get_agent(model_name)])

        logging.info("Judging the responses...")
        judge_results = reward_calculator.get_score(jailbreak_response)

        # Merge attack results and scores
        result = jailbreak_response
        for i in range(len(jailbreak_response)):
            result[i]["judges"] = judge_results[i]

        # Save the results to the jsonl file
        logging.info(f"Saving results to {save_path}...")
        with open(save_path, "a", encoding="utf-8") as f:
            for res in result:
                safe_res = safety_check(res)
                f.write(json.dumps(safe_res, ensure_ascii=False) + "\n")

def handle_evaluate_phase(args, redteam_data, redteam_model, rt_config, reward_calculator, model_name, save_path): 
    """Handle the evaluation phase logic."""
    logging.info(f"Evaluating model using redteam data: {args.redteam_file}")
    
    # Calculate total batch count
    total_batches = len(redteam_data) // rt_config.batch_size + (1 if len(redteam_data) % rt_config.batch_size > 0 else 0)
    
    evaluation_save_path = save_path.replace(".jsonl", "-evaluation.jsonl")

    # Clear the evaluation file first (optional, if rerunning)
    open(evaluation_save_path, "w", encoding="utf-8").close()

    for batch_idx in tqdm.tqdm(range(total_batches), desc="Evaluating batches"):
        batch_start = batch_idx * rt_config.batch_size
        batch_end = min((batch_idx + 1) * rt_config.batch_size, len(redteam_data))
        batch_data = redteam_data[batch_start:batch_end]
        
        logging.info(f"Launching attacks for batch {batch_idx+1}/{total_batches}...")
        jailbreak_response = redteam_model.launch_attacks(batch_data, [rt_config.get_agent(model_name)])
        
        judge_results = reward_calculator.get_score(jailbreak_response)

        for i in range(len(jailbreak_response)):
            jailbreak_response[i]["judges"] = judge_results[i]
        
        # Save only current batch results to avoid duplication
        with open(evaluation_save_path, "a", encoding="utf-8") as eval_file:
            for eval_res in jailbreak_response:
                eval_file.write(json.dumps(eval_res, ensure_ascii=False) + "\n")

    logging.info(f"All evaluation results saved to {evaluation_save_path}.")

            
def handle_redteam_phase(dataloader, redteam_model, save_path):
    """Handle the redteam phase logic."""
    logging.info(f"Redteam Phase: Generating prompts for {len(dataloader)} batches.")
    for batch in tqdm.tqdm(dataloader, desc="Generating Red Team Prompts"):
        generated_prompt = redteam_model.generate(batch['instruction'])
        
        with open(save_path, "a", encoding="utf-8") as f:
            for prompt in generated_prompt:
                f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

def attack(args):
    """Execute the attack task."""
    if args.phase == "evaluate" and not os.path.exists(args.redteam_file):
        raise ValueError("--redteam_file is required for phase `evaluate`")

    for model_name in args.models:
        # Set up task ID and configurations
        task_id = f"{BASE_TASK_ID}/{model_name}"
        
        # judge_config = JudgeConfig(TASK_ID=task_id, model_name="deepseek-r1-250120")
        judge_config = JudgeConfig(TASK_ID=task_id, 
                                #    model_name="deepseek-r1:32b", 
                                #    model_name="qwq:latest",
                                #    base_url=config.OLLAMA_API_BASE_2, 
                                #    api_key=config.OLLAMA_API_KEY_2
                                    # model_name = "qwen-turbo-latest",
                                    base_url=config.DASHSCOPE_API_BASE,
                                    api_key=config.DASHSCOPE_API_KEY,
                                    # model_name = "deepseek-r1"
                                    model_name = "qwen-plus"
                                   )
        
        reward_calculator = RewardCalculator(judge_config)
        
        if args.phase == "evaluate":
            rt_config = RedteamConfig(
                TASK_ID=task_id,
                task_type="evaluate-api",
                device_map="auto",
                batch_size=args.batchsize
            )
            
            redteam_model = RedTeamModel(rt_config)
            redteam_model.setup_model()
            
        else:
            rt_config = RedteamConfig(
                TASK_ID=task_id,
                task_type="redteam-local",
                device_map="auto",
                batch_size=args.batchsize
            )

            # Load the Red Team model
            redteam_model = load_redteam_model(args, rt_config)

        # Generate the save path
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        save_path = generate_save_path(args, model_name, args.phase, timestamp)

        # Load dataset and create dataloader
        dataset = PromptLoader([args.dataset_name])
        dataloader = DataLoader(dataset, batch_size=rt_config.batch_size, shuffle=True, num_workers=4)

        # Handle different attack phases
        if args.phase == "full":
            logging.info(f"Full Attack model: {model_name}")
            handle_full_phase(dataloader, redteam_model, rt_config, reward_calculator, model_name, save_path)

        elif args.phase == "evaluate":
            # load data from jsonl
            redteam_data = []
            with open(args.redteam_file, "r", encoding="utf-8") as f:
                for line in f:
                    redteam_data.append(json.loads(line))

            # redteam_data = redteam_data[:150]
            logging.info(f"Evaluating model: {model_name}")
            handle_evaluate_phase(args, redteam_data, redteam_model, rt_config, reward_calculator, model_name, save_path)

        elif args.phase == "redteam":
            logging.info(f"Redteam Phase for model: {model_name}")
            handle_redteam_phase(dataloader, redteam_model, save_path)

        else:
            raise ValueError(f"Invalid phase: {args.phase}")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Red Team Attack Execution")
    
    parser.add_argument("--models", type=str, nargs="+", default=[
        "gpt-4o-mini",
        # "gpt-4o",
        # "gemini-gemini-1.5-flash",
        # "my-claude-3-haiku-20240307",
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
        # "mistral-7b-instruct-v0.2"
        # "qwq"
        # "deepseek-r1-distill-llama-8b"
        # "deepseek-v3-241226"
        # "meta-llama/Llama-3.3-70B-Instruct"
        # "llama2"
        # "qwen-turbo"
        # "deepseek-v3-241226"
        # "llama3.1:8b"
        # "deepseek-r1-250120",
        # "gemini-1.5-flash",
        # "qwen-turbo",
        # "qwen-max",
        # "claude-3-haiku-20240307",
        # "gpt-4o-mini",
        # "deepseek-v3-250324",
        # "o1-mini",
        # "o3-mini",
        # "o4-mini",
        # "llama2:13b",
        # "qwen-plus",
        # "deepseek-ai/DeepSeek-V3",
        # "llama2:13b-chat-q2_K",
        # "siliconflow-QwQ-32B-Preview",
        # "gpt-4.1",
        # "deepseek-r1-distill-qwen-32b",
        # "gpt-4.1",
        # "qwq:latest",
        # "llama2:70b-chat-q2_K",
        # "llama2:70b-chat-q2_K",
        # "claude-3-haiku-20240307",
        # "llama3.3:70b-instruct-q2_K"
        # "llama2"
        # "vicuna:7b",
        # "vicuna:13b"
        # "qwen:14b",
        # "qwen:7b",
        # "meta/llama-4-maverick-17b-128e-instruct",
        # "deepseek-r1-distill-qwen-7b-250120",
        # "deepseek-r1-distill-qwen-32b-250120",
        # "deepseek-r1-distill-llama-8b",
        # "deepseek-r1-distill-llama-70b",
        # "mistralai/mistral-7b-instruct-v0.2",
        # "meta/llama2-70b",
        # "deepseek-r1-distill-llama-70b",
        # "mistralai/mixtral-8x7b-instruct-v0.1",
        # "llama2:70b",
        # "llama2:13b",
        ]
    )
    
    parser.add_argument("--dataset_name", type=str, default="harmbench", help="Name of the dataset to use")
    parser.add_argument("--adapter_model_path", type=str, default="YOUR_PATH_TO/rl_lora_model", help="Path to the adapter model")
    parser.add_argument("--phase", type=str, default="evaluate", help="Phase of the attack (default: redteam)", choices=["redteam", "evaluate", "full"])
    parser.add_argument("--redteam_file", type=str, default="YOUR_PATH_TO/harmbench.jsonl", help="Path to the redteam file")
    parser.add_argument("--batchsize", type=int, default=10)
    
    args = parser.parse_args()

    setup_logging("logs")

    attack(args)