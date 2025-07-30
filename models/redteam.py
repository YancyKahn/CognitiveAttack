import os
import re
import json
import random
import string
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Deque, Any, Union

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from fastchat.conversation import get_conv_template

from models.language_models import LLM, build_request
from cfgs.base_config import RedteamConfig

from utils.data_tools import safety_check
import math
from concurrent.futures import ThreadPoolExecutor
import ast

class RedTeamModel:
    """Redteam Model Base Class"""
    def __init__(self, config: RedteamConfig):
        self.config = config
        self.redteam_type = config.task_type.split('-')[-1]
        # self.setup_model()
        
        self.bias_strategy = pd.read_csv(self.config.bias_strategy_path, encoding="ISO-8859-1")
        # convert Cognitive Bias to lower
        self.bias_strategy["Cognitive_Bias"] = self.bias_strategy["Cognitive_Bias"].str.lower()
        
    def get_bias_description(self, bias_names: List[str]):
        description_str = ""
        for i, bias in enumerate(bias_names):
            desc = self.bias_strategy[self.bias_strategy["Cognitive_Bias"] == bias]["Short_Description"].values[0]
            description_str += f"""* {bias} : {desc}\n"""
        
        return description_str
    
    def setup_model(self, redteam_model=None, redteam_tokenizer=None):
        self.victim_model = LLM(
            task_type="victim",
            log_path=self.config.log_dir,
            debug=False,
        )
        
        if self.redteam_type == 'api':
            print("Loading API model...")
            self.model = LLM(log_path=self.config.log_dir,
                             task_type=self.config.task_type)
        
        elif self.redteam_type == "local":
            print("Loading local model...")
            if redteam_model is None or redteam_tokenizer is None:
                raise ValueError("redteam_model and redteam_tokenizer must be provided for local model.")
            self.model = redteam_model
            self.tokenizer = redteam_tokenizer
        else:
            raise ValueError(f"Invalid redteam_type in {self.config.task_type}")
            

    def generate_variants_api(self, prompts: List[str], model: LLM=None) -> List[dict]:
        """Using API to generate prompt variants"""
        if model is None:
            api_model = LLM(log_path=self.config.log_dir,
                             task_type=self.config.task_type)
        
        requests = []
        
        for prompt in prompts:
            built_prompt = self.config.system_prompt.replace("[instruction]", prompt)
            
            requests.append(
                build_request(
                    prompt=built_prompt,
                    model_name=self.config.model_name,
                    meta=self.config.to_dict()
                )
            )
        
        raw_results = api_model.generate(requests, max_workers=self.config.max_workers)
        responses = []
        
        result_map = {res["custom_id"]: res for res in raw_results}
        for req in requests:
            result = result_map.get(req["custom_id"], None)
            idx = requests.index(req)
            
            if result and "response" in result and "choices" in result["response"]:
                content = result["response"]["choices"][0]["message"]["content"]
                reasoning_content = result["response"]["choices"][0]["message"]["reasoning_content"]
            else:
                content = "ERROR: Response not available"
                reasoning_content = ""
            
            original_prompt = prompts[idx]
            redteam_input = req["body"]["messages"][0]["content"]
            
            full_response = f"<think>\n{reasoning_content}\n</think>\n{content}"
            responses.append(self._parse_response(content, reasoning_content, original_prompt, full_response, redteam_input))
        
        return responses

    def generate_variants_local(self, prompts: List[str]) -> List[dict]:
        """Use local model to generate prompt variants"""
        convs = []
        openai_messages_convs = []
        for i, prompt in enumerate(prompts):
            conv = get_conv_template(self.config.template_name)
            built_prompt = self.config.system_prompt.replace("[instruction]", prompt)
            conv.append_message(conv.roles[0], built_prompt)
            conv.append_message(conv.roles[1], None)
            convs.append(conv.get_prompt())
            openai_messages_convs.append(conv.to_openai_api_messages())

        with torch.no_grad():
            inputs = self.tokenizer(convs, return_tensors="pt", padding=True).to(self.config.device)
            attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else torch.ones_like(inputs['input_ids'])

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
        
        responses = []
        for i, output in enumerate(outputs):
            original_prompt = prompts[i]
            redteam_input = openai_messages_convs[i][-1]["content"]
            
            input_len = len(inputs.input_ids[i])
            generated = output[input_len:]
            full_response = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            end_index = full_response.find("</think>")
            if end_index != -1:
                reasoning_content = full_response[:end_index].strip()
                content = full_response[end_index:].strip()
            else:
                reasoning_content = ""
                content = full_response.strip()
            
            full_response = f"<think>\n{full_response}"
            responses.append(self._parse_response(content, reasoning_content, original_prompt, full_response, redteam_input))
        
        return responses

    def _parse_response(self, content: str, reasoning_content: str, original_prompt: str, full_response: str, redteam_input: str) -> dict:
        """Extract biases and prompt from content"""
        bias_match = re.search(r"#thebias:\s*\[(.*?)\]", content, re.DOTALL)
        prompt_match = re.search(r'#theprompt:\s*(.+?)(?=\n#|$)', content, re.DOTALL)
        
        try:
            biases_raw = bias_match.group(1) if bias_match else "[]"
            biases = ast.literal_eval(biases_raw)
        except Exception as e:
            print(f"parse bias failed: {e} on `models/redteam.py`")
            biases = []

        # to lower
        biases = [bias.lower() for bias in biases]
        
        return {
            "custom_id": "".join(random.choices(string.ascii_letters + string.digits, k=8)),
            "redteam": {
                "task_type": self.config.to_dict()["task_type"],
                "original_prompt": original_prompt,
                "redteam_input": redteam_input,
                "content": content,
                "reasoning_content": reasoning_content,
                "biases": biases,
                "full_response": full_response,
                "enhanced_prompt": prompt_match.group(1).strip() if prompt_match else original_prompt
            }
        }
    
    def _save_result(self, responses: List[dict], save_name: str="redteam_result.jsonl"):
        """Save the redteam results to a JSONL file"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        with open(os.path.join(self.config.log_dir, save_name), "a+", encoding="utf-8") as f:
            for res in responses:
                f.write(json.dumps(safety_check(res), ensure_ascii=False) + "\n")
    
    def generate(self, prompts: List[str]) -> List[dict]:
        """Generate prompt variants based on the redteam type"""
        if self.redteam_type == "api":
            responses = self.generate_variants_api(prompts)
        elif self.redteam_type == "local":
            responses = self.generate_variants_local(prompts)
        else:
            raise ValueError(f"Invalid redteam_type in {self.config.task_type} `models/redteam.py`")

        self._save_result(responses)
        
        return responses
    
    def generate_mixed(self, prompts: List[str], alpha=0.5) -> List[dict]:
        split_index = math.ceil(len(prompts) * alpha)
    
        prompts_api = prompts[:split_index]
        prompts_local = prompts[split_index:]
        
        with ThreadPoolExecutor() as executor:
            future_api = executor.submit(self.generate_variants_api, prompts_api, None)
            future_local = executor.submit(self.generate_variants_local, prompts_local)
            
            results_api = future_api.result()
            results_local = future_local.result()
        
        self._save_result(results_api + results_local)
        return results_api + results_local
        
    def launch_attacks(self, variants_results: List[dict], agent_list: List=[]) -> List[dict]:
        """Attack the variants using multiple agents"""
        requests = [
            self._build_attack_request(item["redteam"]["enhanced_prompt"], item["custom_id"], agent_list) for item in variants_results
        ]
        
        raw_results = self.victim_model.generate(requests, max_workers=self.config.max_workers)
        
        return self._reconstruct_attack_results(variants_results, raw_results)                          
    def _build_attack_request(self, prompt: str, custom_id: str, agent_list: List=[]) -> dict:
        """Build attack request structure"""
        if agent_list:
            agent = random.choice(agent_list)
        else:
            agent = random.choice(self.config.agents)
        return build_request(
            prompt=prompt,
            model_name=agent.model_name,
            meta=agent.to_dict(),
            custom_id=custom_id
        )
    
    def _parse_attack_response(self, result: dict) -> str:
        """Parse the attack response content"""
        try:
            return result['response']['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return "$ERROR$"
    def _reconstruct_attack_results(self, variants_results: List[dict], results: List[dict]) -> List[dict]:
        """Reconstruct the attack results from the raw results"""
        result_map = {res['custom_id']: res for res in results}
        ordered_results = []
        
        for req in variants_results:
            res = result_map.get(req['custom_id'], None)
            if res and 'response' in res and "choices" in res["response"]:
                ordered_results.append({
                    "custom_id": res['custom_id'],
                    "redteam": req["redteam"],
                    "jailbreak": {
                        "attack_meta": res["request"]['meta'],
                        "original_prompt": req["redteam"]["original_prompt"],
                        "attack_prompt": res["request"]['body']['messages'][0]['content'],
                        "attack_response": self._parse_attack_response(res),
                        "status": "success"
                    }
                })
            else:
                ordered_results.append({
                    "custom_id": res['custom_id'],
                    "redteam": req["redteam"],
                    "jailbreak": {
                        "attack_meta": res["request"]['meta'],
                        "original_prompt": req["redteam"]["original_prompt"],
                        "attack_prompt": req["redteam"]["enhanced_prompt"],
                        "attack_response": "$ERROR$",
                        "status": "error"
                    }
                })
                
        return ordered_results

    def load_redteam_model(self, adapter_model_path=None):
        
        def load_model(model_path: str):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,  
            #     bnb_8bit_compute_dtype=torch.bfloat16  
            # )
            
            
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.config.device_map,
                trust_remote_code=True,
                # load_in_4bit=True
                quantization_config=bnb_config
            )
            
        base_model = load_model(self.config.model_path)
        
        if adapter_model_path is None:
            adapter_model_path = self.config.adapter_model_path

        model = PeftModel.from_pretrained(
            base_model,  # Base model
            model_id=adapter_model_path,  # Path to LoRA adapter weights
            device_map=self.config.device_map
        )
        
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path, trust_remote_code=True)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.eos_token_id
            
        # Set the model to evaluation mode
        model.eval()

        self.setup_model(redteam_model=model, redteam_tokenizer=tokenizer)