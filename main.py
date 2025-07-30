import logging
from cfgs.base_config import JailRLConfig, RedteamConfig, JudgeConfig
from train import CustomPPOTrainer
from utils.data_tools import PromptLoader
from models.redteam import RedTeamModel
from utils.reward_calculator import RewardCalculator
from torch.utils.data import DataLoader
import argparse
import os
from fastchat.conversation import get_conv_template

TASK_ID = "rl-jailbreak-cognitive-bias"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def main(args):
    rl_config = JailRLConfig(TASK_ID=TASK_ID, seed=args.seed)
    rt_config = RedteamConfig(TASK_ID=TASK_ID,task_type="redteam-local", seed=args.seed)
    judge_config = JudgeConfig(TASK_ID=TASK_ID)
    
    custom_trainer = CustomPPOTrainer(rl_config)
    
    redteam_model = RedTeamModel(rt_config)
    redteam_model.setup_model(custom_trainer.red_model, custom_trainer.tokenizer)
    
    reward_calculator = RewardCalculator(judge_config)
    
    dataset = PromptLoader(rl_config.train_datasets)
    dataloader = DataLoader(dataset, 
                            batch_size=rl_config.rl_batch_size,
                            shuffle=True)
    
    
    # offline train
    if rl_config.rl_offline_train:
        custom_trainer.train_offline()
    
    for epoch in range(rl_config.rl_num_epochs):
        for batch in dataloader:
            instructions = batch['instruction']
            
            print(f"STEP: {custom_trainer.current_step} Generating Redteam Prompt...")            

            generated_prompt = redteam_model.generate_mixed(instructions)
            
            print(f"STEP: {custom_trainer.current_step} Launching Attacks ...")

            jailbreak_response = redteam_model.launch_attacks(generated_prompt)
            
            print(f"STEP: {custom_trainer.current_step} Judging ...")

            rewards, safety_rewards, intention_rewards = reward_calculator.calculate_rewards(jailbreak_response)

            valid_samples = []
            
            print(f"STEP: {custom_trainer.current_step} Update Policy ...")
            for redteam, reward, safety_reward, intention_reward in zip(generated_prompt, rewards, safety_rewards, intention_rewards):
                conv = get_conv_template(rl_config.template_name)
                conv.append_message(conv.roles[0], redteam["redteam"]["redteam_input"])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                valid_samples.append({
                    "custom_id": redteam["custom_id"],
                    "prompt": prompt,
                    "response": redteam["redteam"]['full_response'],
                    "rewards": reward,
                    "safety_rewards": safety_reward,
                    "intention_rewards": intention_reward
                })
                   
            for sample in valid_samples:
                custom_trainer.memory.append(sample) 
                  
            custom_trainer.save_memory()

            if len(custom_trainer.memory) >= rl_config.rl_batch_size:
                logging.info("==============Training RL Step==============")
                sampled_batch = custom_trainer.batch_from_memory()
                custom_trainer.train_step(sampled_batch)
                
    # save model
    custom_trainer.save_lora_model()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task_id", type=str, default=TASK_ID)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)
