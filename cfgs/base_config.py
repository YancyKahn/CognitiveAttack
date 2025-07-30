
import config
from cfgs.system_prompt import system_prompt_for_gptjudge, system_prompt_for_intention, system_prompt_combination_bias
import os
import torch 

class BaseConfig:
    """Base Config"""
    
    _sensitive_fields = {"api_key"}
    
    def __init__(self, **kwargs):
        self._set_attrs(kwargs)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if 'device_map' not in kwargs:
            self.device_map = "auto"
            
        if 'seed' not in kwargs:
            self.seed = 42
    def _set_attrs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self) -> dict:
        return vars(self)
    
    def to_safety_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if k not in self._sensitive_fields}
    @classmethod 
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)
    
    
class JudgeConfig(BaseConfig):
    """Safety Judge Config"""
    def __init__(self, TASK_ID, **kwargs):
        defaults = dict(
            base_url=config.VOLCENGINE_API_BASE,
            api_key=config.VOLCENGINE_API_KEY,
            log_dir=f"logs/redteam/{TASK_ID}/judge",
            model_name="deepseek-r1-250120",
            # model_name="deepseek-v3-250324",
            temperature=0.1,
            top_p=1.0,
            max_tokens=2048,
            max_workers=8,
            system_prompt_gptjudge=system_prompt_for_gptjudge(),
            system_prompt_intention=system_prompt_for_intention(),
            task_type="judge",
        )
        super().__init__(**{**defaults, **kwargs})

class VictimAgentConfig(BaseConfig):
    """Victim Model Config"""
    def __init__(self, **kwargs):
        required = ['base_url', 'api_key', 'model_name']
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
            
        defaults = dict(
            task_type=f"victim"
        )
        super().__init__(**defaults, **kwargs)


# cfgs/base_config.py
class RedteamConfig(BaseConfig):
    def __init__(self, TASK_ID, **kwargs):
        defaults = dict(
            base_url=config.VOLCENGINE_API_BASE,
            api_key=config.VOLCENGINE_API_KEY,
            model_name="deepseek-r1-250120",
            model_path=config.JAILRL_MODEL,
            template_name=config.TEMPLATE,
            log_dir=f"logs/redteam/{TASK_ID}/victim",
            bias_strategy_path="/mnt/data2/yxk/cognitive-attack/data/cognitive-bias-taxonomy.csv",
            max_workers=8,
            temperature=0.9,
            top_p=0.9,
            max_tokens=4096,
            system_prompt=system_prompt_combination_bias(),
            task_type="redteam-local", # redteam-api
            device_map = {0:"cuda:1"},
            adapter_model_path=None,
            batch_size=8,
            
            generate_type="raw"
        )

        super().__init__(**{**defaults, **kwargs})
        
        self.agents = [
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="deepseek-v3-250324",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="deepseek-v3-241226",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="deepseek-r1-250120",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="doubao-1-5-lite-32k-250115",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="mistral-7b-instruct-v0.2",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="deepseek-r1-distill-qwen-7b-250120",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.VOLCENGINE_API_BASE,
                    api_key=config.VOLCENGINE_API_KEY,
                    model_name="deepseek-r1-distill-qwen-32b-250120",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="qwen-turbo",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-v3",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-r1",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-r1-distill-llama-8b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-r1-distill-llama-70b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-r1-distill-qwen-7b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="deepseek-r1-distill-qwen-32b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="llama3.3-70b-instruct",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="llama-4-scout-17b-16e-instruct",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="qwq-32b-preview",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="qwen-plus",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.DASHSCOPE_API_BASE,
                    api_key=config.DASHSCOPE_API_KEY,
                    model_name="qwen-max",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="gpt-4o-mini",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="gpt-4o-mini-2024-07-18",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="o3-mini",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="o1-mini",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="o4-mini",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=2048
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="gpt-4o",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="gpt-4.1",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    model_name="claude-3-haiku-20240307",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.SILICONFLOW_API_BASE,
                    api_key=config.SILICONFLOW_API_KEY,
                    model_name="meta-llama/Llama-3.3-70B-Instruct",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.SILICONFLOW_API_BASE,
                    api_key=config.SILICONFLOW_API_KEY,
                    model_name="deepseek-ai/DeepSeek-V3",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="llama2",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="llama3",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="llama3.1:8b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="llama3.2",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="llama3.3:70b-instruct-q2_K",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="vicuna:13b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="vicuna:7b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="qwen:7b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE,
                    api_key=config.OLLAMA_API_KEY,
                    model_name="qwen:14b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_1,
                    api_key=config.OLLAMA_API_KEY_1,
                    model_name="llama2:13b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="llama2:latest",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_1,
                    api_key=config.OLLAMA_API_KEY_1,
                    model_name="llama2:70b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="deepseek-r1:32b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="deepseek-r1:32b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=1024
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="qwen2.5:7b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="qwen2.5:14b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="qwq:latest",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="llama2:13b-chat-q2_K",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.OLLAMA_API_BASE_2,
                    api_key=config.OLLAMA_API_KEY_2,
                    model_name="llama2:70b-chat-q2_K",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="meta/llama-4-maverick-17b-128e-instruct",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="meta/llama-4-scout-17b-16e-instruct",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="mistralai/mistral-7b-instruct-v0.2",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="deepseek-ai/deepseek-r1",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="mistralai/mixtral-8x7b-instruct-v0.1",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.NV_API_BASE,
                    api_key=config.NV_API_KEY,
                    model_name="meta/llama2-70b",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                VictimAgentConfig(
                    base_url=config.ONE_API_BASE,
                    api_key=config.ONE_API_KEY,
                    model_name="gemini-1.5-flash",
                    temperature=0.9,
                    top_p=0.9,
                    max_tokens=512
                ),
                ]
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "agents": [agent.to_safety_dict() for agent in self.agents]
        }

    def to_safety_dict(self):
        return {
            **super().to_safety_dict(),
            "agents": [agent.to_safety_dict() for agent in self.agents]
        }
        
    def get_agent(self, agent_name):
        agent_map = {agent.model_name: agent for agent in self.agents}
        if agent_name not in agent_map:
            raise ValueError(f"Agent {agent_name} not found in list {', '.join([agent.model_name for agent in self.agents])}.")
        return agent_map[agent_name]
        

class JailRLConfig(BaseConfig):
    def __init__(self, TASK_ID, **kwargs):
        defaults = dict(
            # Model Config
            model_path=config.JAILRL_MODEL,
            template_name=config.TEMPLATE,
            adapter_model_path=None,
            task_type="jail-rl",
            
            # Task Config
            rl_offline_train=False,
            num_strategies=154,
            max_strategies=3,   # 最大选择策略数
            memory_size=1000000,   # 记忆库大小
            
            # Data Config
            bias_strategy_path="data/cognitive-bias-taxonomy.csv",
            train_datasets=["pair", "hexphi"],
            
            # TRL PPO training Config
            rl_learning_rate=3e-5,              
            rl_batch_size=16,                   
            rl_mini_batch_size=2,               
            rl_optimize_cuda_cache=True,        
            rl_adap_kl_ctrl=True,               
            rl_init_kl_coef=0.2,                
            rl_kl_penalty="kl",                 
            rl_vf_coef=0.1,                     
            rl_cliprange=0.2,                   
            rl_cliprange_value=0.2,             
            rl_rollout_batch_size=None,         
            rl_early_stopping=False,            
            rl_log_with='wandb',                        
            wandb_project_name='Cognitive-Redteam',     
            wandb_run_name=TASK_ID,                     
            
            # RL Training Config
            rl_num_epochs=20,
            rl_save_freq=20,                   
            
            # LoRA Config
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_bias="none",
            lora_task_type="CAUSAL_LM",
            
            # Device Config
            device = "cuda",
            
            # Path Config
            log_dir = f"logs/redteam/{TASK_ID}/RL-memory",
            output_dir = f"logs/redteam/{TASK_ID}/checkpoints",
            wandb_dir = f"logs/redteam/{TASK_ID}/wandb"
        )
        
        super().__init__(**defaults, **kwargs)
        
        # Wandb setting offline model
        if self.rl_log_with == "wandb":
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = self.wandb_dir