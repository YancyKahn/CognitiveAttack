from typing import List
import pandas as pd
import json
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

from copy import deepcopy
from typing import Dict, Any
def safety_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """返回一个不包含任何 `api_key` 的新字典（不影响原始数据）"""
    def filter_api_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            # 过滤掉 key 包含 'api_key' 的项，并递归处理剩余值
            return {
                k: filter_api_keys(v)
                for k, v in obj.items()
                if not (isinstance(k, str) and 'api_key' in k.lower())
            }
        elif isinstance(obj, list):
            # 递归处理列表中的每个元素
            return [filter_api_keys(item) for item in obj]
        else:
            # 其他类型（str, int, bool 等）直接返回
            return obj

    return filter_api_keys(deepcopy(data))  # 深拷贝确保不影响原始数据


def custom_collate_fn_cold_start(batch, pad_token_id):
    # 动态计算最大长度
    max_len = max(len(x['input_ids']) for x in batch)
    
    # 初始化填充容器
    padded_inputs = {
        'input_ids': [],
        'attention_mask': [],
        'biases': []
    }
    
    # 填充每个样本
    for item in batch:
        # 计算需要填充的长度
        pad_len = max_len - len(item['input_ids'])
        
        # 填充input_ids和attention_mask
        padded_inputs['input_ids'].append(
            torch.cat([
                item['input_ids'], 
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        )
        padded_inputs['attention_mask'].append(
            torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        padded_inputs['biases'].append(item['biases'])
        
    return {
        'input_ids': torch.stack(padded_inputs['input_ids']),
        'attention_mask': torch.stack(padded_inputs['attention_mask']),
        'biases': padded_inputs['biases']
    }

def custom_collate_fn_rl(batch, pad_token_id):
    """
    自定义批处理函数，确保张量大小一致并处理目标偏见集合
    """
    # 对 input_ids 和 attention_mask 进行填充
    input_ids = pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id
    )
    attention_mask = pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0  # attention_mask 的填充值为 0
    )
    
    instructions = [item['instruction'] for item in batch]
    # 返回一个字典，包含所有需要的字段
    return {
        'instructions': instructions,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

class ColdStartDataset(Dataset):
    """冷启动数据集"""
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = self.tokenizer(
            item['instruction'],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'instruction': item['instruction'],
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'biases': item['biases']  # 目标偏见集合
        }

class PromptLoader(Dataset):
    def __init__(self, datasets: List=["pair"]):
        self.data = self.load_data(datasets)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'instruction': item['instruction'],
        }
        
    def get_batch(self, batch_size: int):
        return self.data.sample(batch_size)["goal"].tolist()

    def load_data(self, dataset_names):
        datas = {
            "pair": {
                "path": "data/harmful/pair/harmful_behaviors_custom.csv",
                "key": "instruction"
            },
            "advbench": {
                "path": "data/harmful/advbench/harmful_behaviors.csv",
                "key": "instruction"
            },
            "beavertails": {
                "path": "data/harmful/BeaverTails/beavertails_30k_train.csv",
                "key": "prompt"
            },
            "hexphi": {
                "path": "data/harmful/HEx_PHI/HExPHI.csv",
                "key": "instruction"
            },
            "hexphi-sub10": {
                "path": "data/harmful/HEx_PHI/HEx_PHI_sub10.json",
                "key": "data"
            },
            "harmbench": {
                "path": "/mnt/data2/yxk/cognitive-attack/build/HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv",
                "key": "Behavior"
            },
            "IJCAI": {
                "path": "/mnt/data2/yxk/cognitive-attack/data/IJCAI/IJCAI.csv",
                "key": "修改命题"
            }
        }
        
        datasets = pd.DataFrame()
        
        for name in dataset_names:
            path = datas[name]["path"]
            
            if path.endswith(".csv"):
                if "IJCAI" == name:
                    df = pd.read_csv(path, encoding="gb2312")
                else:
                    df = pd.read_csv(path)
            elif path.endswith(".jsonl"):
                df = pd.read_json(path, lines=True)
            elif path.endswith(".json"):
                if name == "hexphi-sub10":
                    df = pd.read_json(path)
                    df = df.explode('data', ignore_index=True)
            else:
                raise ValueError("Unsupported file format in `utils/data_loader.py`")

            df['dataset'] = name
            if name == "beavertails":
                df = df[df["is_safe"] == True]
                df = df.sample(1000)
            
            if name == "hexphi":
                df = df.sample(frac=0.5)
            
            datasets = pd.concat([datasets, df], ignore_index=True)
            
            # convert key column to instruction
            datasets = datasets.rename(columns={datas[name]["key"]: "instruction"})
            
        datasets = datasets.to_dict('records')
        
        print(f"Successfully loaded data. Total records: {len(datasets)}")
        return datasets
    