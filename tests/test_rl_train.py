import sys
sys.path.append(".")

from cfgs.base_config import JailRLConfig
from train import CustomPPOTrainer


if __name__ == "__main__":
    rl_config = JailRLConfig(TASK_ID="rl-jailbreak-cognitive-bias")
    
    # 初始化模块
    trainer = CustomPPOTrainer(rl_config)
    trainer.load_memory()
    
    print("length of memory: ", len(trainer.memory))
    sampled_batch = trainer.batch_from_memory()
    trainer.train_step(sampled_batch)
