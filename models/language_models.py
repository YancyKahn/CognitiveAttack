import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import string
import os
import random
from functools import partial
import json
import config
from utils.data_tools import safety_check

def build_request(prompt, model_name, meta, custom_id=None):
    request = {
        "custom_id": "".join(random.choices(string.ascii_letters + string.digits, k=8)) if custom_id is None else custom_id,
        "meta": meta,
        "body": {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt
            }
            ]
        }
    }

    return request



class LLM:
    def __init__(self, log_path="log/tmp", task_type="tmp", debug=False):
        self.log_path = log_path
        self.debug = debug
        self.task_type = task_type
        
        if not self.debug:
            self.save_path = os.path.join(self.log_path, f"{task_type}_result.jsonl")
            log_file_path = os.path.join(self.log_path, f"{task_type}_run_logs.log")

            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                filename=log_file_path,
                filemode='w'
            )

    def call_openai(self, request):
        client = OpenAI(api_key=request["meta"]["api_key"], base_url=request["meta"]["base_url"])
        response = client.chat.completions.create(
            model=request["body"]["model"],
            messages=request["body"]["messages"],
            max_tokens=request["meta"].get("max_tokens", 2048),
        )
        result = {}
        result["custom_id"] = request["custom_id"]
        result["request"] = request.copy()
        result["response"] = response.to_dict()
        return result

    def call_with_retry(self, request, retries=5):
        logging.info(f'Executing model {self.task_type}---{request["body"]["model"]}')
        for attempt in range(retries):
            try:
                return self.call_openai(request)
            except Exception as e:
                if not self.debug:
                    logging.warning(
                        f"Error on attempt {attempt + 1} for request {request['custom_id']} - {request['body']['model']}, "
                        f"error: {e}"
                    )
                    if "RPH limit reached." in e.args[0]:
                        logging.warning(
                            f"Hour Rate limit reached, sleeping for 1 hour")
                        time.sleep(3600)
                    elif "RPM limit reached." in e.args[0]:
                        logging.warning(
                            f"Minute Rate limit reached, sleeping for 1 minute")
                        time.sleep(60)
                    else:
                        time.sleep(2 ** attempt)
                else:
                    print(
                        f"Error on attempt {attempt + 1} for request {request['custom_id']} - {request['body']['model']}, "
                        f"error: {e}")
                    time.sleep(2 ** attempt)
                    
            raise Exception(
                f"Failed after {retries} attempts for request {request['custom_id']} - {request['body']['model']}")

    def generate(self, requests, max_workers=3):
        if not self.debug:
            logging.info(f"Starting {self.task_type} process with requests")
        else:
            print(f"Starting {self.task_type} process with requests")

        max_workers = min(len(requests), max_workers)
        call_with_retry_fn = partial(self.call_with_retry)

        generated_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(call_with_retry_fn, req): i for i, req in enumerate(requests)}

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = safety_check(future.result())
                    generated_list.append(result)
                    
                    if not self.debug:
                        logging.info(f"Request {index} completed successfully, model: {result['custom_id']}-{result['request']['body']['model']}")
                        # logging.info(f"Response: {result["response"]["choices"][0]["message"]["content"]}")

                        with open(self.save_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    else:
                        print(f"Request {index} completed successfully, model: {result['custom_id']}-{result['request']['body']['model']}")
                        print(f"=================================={self.task_type}=======================================\n",
                              f"{result['response']['choices'][0]['message']['content']}")
                        
                except Exception as e:
                    if not self.debug:
                        logging.error(f"Request at index {index} failed, error: {e}")
                    else:
                        print(f"Request at index {index} failed, error: {e}")
                    
                    result = {}
                    result["custom_id"] = requests[index]["custom_id"]
                    result["request"] = requests[index].copy()
                    result["response"] = "$ERROR$"
                    
                    generated_list.append(result)

        if not self.debug:
            logging.info(f"Generate {self.task_type} process completed")
        else:
            print(f"Generate {self.task_type} process completed")
        
        return generated_list
        



if __name__ == "__main__":
    llm = LLM(debug=True)
    
    # requests = [build_request("Hello, how are you?", "deepseek-v3-241226", {"max_tokens": 100, "api_key": config.VOLCENGINE_API_KEY, "base_url": config.VOLCENGINE_API_BASE})]
    requests = [build_request("Hello, how are you?", "gpt-4o-mini", {"max_tokens": 100, "api_key": config.OPENAI_API_KEY, "base_url": config.OPENAI_API_BASE})]
    
    llm.generate(requests)