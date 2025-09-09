import os
from typing import List, Union
import json
import numpy as np
import math
import base64
import json
import time
import datetime
from tqdm import tqdm
import pandas as pd
from pyvent.tools.save_tools import SaveTool
import asyncio
import concurrent.futures

from openai import AzureOpenAI, RateLimitError, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
import tiktoken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from pyvent.constants import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_BATCH_OPENAI_KEY,
    AZURE_BATCH_OPENAI_ENDPOINT,
)


MODEL_MAPPING = {"gpt-4o-batch": "gpt-4o-2", "gpt-4o-mini-batch": "gpt-4o-mini-2", "o3-batch": "o3", "o3":"o3-2"}


def bool_output_parser(output: Union[str, bool]) -> bool:
    if isinstance(output, bool):
        return output

    if not output:
        return False

    if not isinstance(output, str):
        return False

    if "true" in output.lower():
        return True
    else:
        return False


MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4o-batch", "gpt-4o-mini-batch", "o1", 'gpt-4.1', 'o3', 'o3-batch']

MAX_TOKEN_DICT = {"text-embedding-3-large": 8191}

MODEL_PRICING = {
    "gpt-4o": {"prompt": 2.5 / 1000000, "completion": 10 / 1000000},
    "gpt-4o-mini": {"prompt": 0.15 / 1000000, "completion": 0.6 / 1000000},
    "gpt-4o-mini-batch": {
        "prompt": 0.15 / 1000000 / 2,
        "completion": 0.6 / 1000000 / 2,
    },
    "gpt-4o-batch": {"prompt": 2.5 / 1000000 / 2, "completion": 10 / 1000000 / 2},
    "o1": {"prompt": 15 / 1000000 / 2, "completion": 60 / 1000000 / 2},
    "gpt-4.1": {"prompt": 2.5 / 1000000 / 2, "completion": 10 / 1000000 / 2},
    "o3-2": {"prompt": 2.5 / 1000000, "completion": 10 / 1000000},
    "o3": {"prompt": 15 / 1000000 / 2, "completion": 60 / 1000000 / 2},
}


def bool_output_parser(output: str) -> bool:
    if "true" in output.lower():
        return True
    else:
        return False


class OpenAIAPI:
    def __init__(self, model: str = "gpt-4o", chunk_size: int = 32, saver: SaveTool = None):
        if model not in MODELS:
            raise ValueError(f"Model {model} not supported. Supported models: {MODELS}")

        self.model: str = MODEL_MAPPING.get(model, model)
        self.chunk_size = chunk_size

        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        self.total_cost: float = 0.0  # Running cost of all inferences

        api_version = "2024-12-01-preview"

        self.client = AzureOpenAI(
            api_key=AZURE_BATCH_OPENAI_KEY, api_version=api_version, azure_endpoint=AZURE_BATCH_OPENAI_ENDPOINT
        )

        self.async_client = AsyncAzureOpenAI(
            api_key=AZURE_BATCH_OPENAI_KEY, api_version=api_version, azure_endpoint=AZURE_BATCH_OPENAI_ENDPOINT
        )

        self.batch_client = AzureOpenAI(
            api_key=AZURE_BATCH_OPENAI_KEY, api_version=api_version, azure_endpoint=AZURE_BATCH_OPENAI_ENDPOINT
        )

        self.saver = saver if saver else SaveTool(data_path="data/openai", input_path="input", output_path="output")

    def get_cost(self) -> float:
        return self.total_cost

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def get_predicted_total_cost(self, prompts, estimated_output_tokens: int = None, average_output_phrase: str = None):
        import random
        total_prompt_tokens = 0
        # randomly sample 1/10 of the prompts
        prompts = random.sample(prompts, len(prompts)//10)
        for prompt in tqdm(prompts, desc="Calculating prompt tokens"):
            for msg in prompt:
                content_tokens = len(self.encoding.encode(msg["content"]))
                content_tokens += 4
                total_prompt_tokens += content_tokens
            # every message primed with 3 extra tokens for assistant
            total_prompt_tokens += 3
        # multiply by 10 to account for the 1/10 sample
        total_prompt_tokens *= 10
        estimated_output_tokens = (
            estimated_output_tokens if estimated_output_tokens else len(self.encoding.encode(average_output_phrase))
        ) * 10
        estimated_output_tokens *= len(prompts)
        total_prompt_token_cost = total_prompt_tokens * MODEL_PRICING[self.model]["prompt"]
        estimed_completion_token_cost = estimated_output_tokens * MODEL_PRICING[self.model]["completion"]

        print(f"""
        Total prompt tokens: {total_prompt_tokens}
        Estimated output tokens: {estimated_output_tokens}
        Prompt token cost: ${total_prompt_token_cost:.4f}
        Estimated completion token cost: ${estimed_completion_token_cost:.4f}
        Estimated Total cost: ${total_prompt_token_cost + estimed_completion_token_cost:.4f}
        """)

    def calc_cost(self, model: str, response: ChatCompletion) -> float:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        prompt_cost = MODEL_PRICING[model]["prompt"]
        completion_cost = MODEL_PRICING[model]["completion"]
        return (prompt_tokens * prompt_cost) + (completion_tokens * completion_cost)

    def cosine_similarity(self, v1: List[float], v2: Union[List[List[float]], List[float]]) -> List[float]:
        if isinstance(v2[0], list):
            return [np.dot(v1, v2[i]) / (np.linalg.norm(v1) * np.linalg.norm(v2[i])) for i in range(len(v2))]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def embeddings(self, text: Union[List[str], str]) -> json:
        def truncate(text: str):
            encoded_text = self.encoding.encode(text)
            encoded_text = encoded_text[: MAX_TOKEN_DICT["text-embedding-3-large"]]
            return self.encoding.decode(encoded_text)

        truncated_text = None
        if isinstance(text, str):
            truncated_text = truncate(text)
        else:
            truncated_text = [truncate(t) for t in text]

        response = self.client.embeddings.create(model="text-embedding-3-large", input=truncated_text)
        if isinstance(text, str):
            return response.data[0].embedding
        else:
            return [embedding.embedding for embedding in response.data]

    def divide_chunks(self, prompt_list: list, chunk_size: int):
        return [prompt_list[i : i + chunk_size] for i in range(0, len(prompt_list), chunk_size)]

    @retry(
        wait=wait_random_exponential(min=5, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _chat_completion(self, prompt, **model_kwargs):
        res = await self.async_client.chat.completions.create(messages=prompt, model=self.model, **model_kwargs)
        return res

    async def run_async(self, prompts, **model_kwargs):
        """
        Run the agent asynchronously
        Args:
            prompts: takes the format of [[SystemMessage, HumanMessage], ...]
        """

        chunks = list(self.divide_chunks(prompts, self.chunk_size))

        async def process_chunk(chunk) -> List[ChatCompletion]:
            tasks = []
            for prompt in chunk:
                tasks.append(self._chat_completion(prompt, **model_kwargs))
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            parsed_responses = []
            for response in responses:
                if isinstance(response, Exception):
                    # exception_text = str(response)
                    parsed_responses.append(response)
                else:
                    parsed_responses.append(response)
            return parsed_responses

        self.responses = []
        for chunk in (pbar := tqdm(chunks, total=len(chunks), unit="chunk")):
            pbar.set_description(f"Running cost ${self.total_cost:.4f}")
            responses = await process_chunk(chunk)
            for response in responses:
                if response and isinstance(response, ChatCompletion):
                    self.total_cost += self.calc_cost(self.model, response)
                    self.responses.append(response.choices[0].message.content)
                else:
                    self.responses.append(response)
        return self.responses

    # Batch Functionality

    def save_batch_files(self, prompts, **model_kwargs) -> List[str]:
        batch_output = []
        for i, prompt in enumerate(prompts):
            base = {
                "custom_id": None,
                "method": "POST",
                "url": "/chat/completions",
                "body": {"model": self.model, "messages": [], **model_kwargs},
            }
            base["custom_id"] = str(i)
            base["body"]["messages"] = prompt
            batch_output.append(base)

        split_batches = np.array_split(batch_output, np.ceil(len(batch_output) / 30000))
        file_paths = []
        for i, batch in enumerate(split_batches):
            filepath = os.path.join(self.saver.input_path, f"prompt_batch_{i}.jsonl")
            with open(filepath, "w") as f:
                for item in batch:
                    f.write(json.dumps(item) + "\n")
            file_paths.append(filepath)
        return file_paths

    def submit_batch(self, file_paths: List[str]) -> List[str]:
        # upload files
        file_ids = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.batch_client.files.create, file=open(file_path, "rb"), purpose="batch"): file_path for file_path in file_paths}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading files"):
                file = future.result()
                file_id = file.id
                file_ids.append(file_id)
        
        file_ids = [future.result().id for future in futures]

        status_dict = {k: "running" for k in file_ids}
        while any([status_dict[k] not in ("processed") for k in file_ids]):
            for file_id in file_ids:
                file_response = self.batch_client.files.retrieve(file_id)
                status = file_response.status
                status_dict[file_id] = status
                print(f"{datetime.datetime.now()} File Id: {file_id},  Status: {status}")
            time.sleep(20)
        print(f"Files uploaded with ids: {file_ids}")

        batch_ids = []
        for file_id in file_ids:
            batch_response = self.batch_client.batches.create(
                input_file_id=file_id,
                endpoint="/chat/completions",
                completion_window="24h",
            )
            batch_id = batch_response.id
            batch_ids.append(batch_id)
        return batch_ids

    def download_batch(self, batch_id: str) -> List[dict]:
        batch_response = self.batch_client.batches.retrieve(batch_id)
        output_file_id = batch_response.output_file_id
        file_response = self.batch_client.files.content(output_file_id)
        output_file_path = os.path.join(self.saver.output_path, f"{batch_id}.jsonl")
        with open(output_file_path, "wb") as f:
            f.write(file_response.content)
        raw_responses = file_response.text.strip().split("\n")
        json_output = []
        for raw_response in raw_responses:
            json_response = json.loads(raw_response)
            json_output.append(json_response)
        return json_output

    def retrieve_batches(self, batch_ids: List[str]) -> List[dict]:
        status = "validating"
        status_dict = {k: status for k in batch_ids}
        while any([status_dict[k] not in ("completed", "failed", "canceled") for k in batch_ids]):
            for batch_id in batch_ids:
                batch_response = self.batch_client.batches.retrieve(batch_id)
                status = batch_response.status
                status_dict[batch_id] = status
                print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")
            time.sleep(90)

        all_responses = []
        for batch_id, status in status_dict.items():
            if status == "completed":
                print(f"Batch {batch_id} completed")
                responses = self.download_batch(batch_id)
                all_responses.extend(responses)
                print(f"Retrieved: {len(responses)}")
            elif status == "failed":
                print(f"Batch {batch_id} failed")
            elif status == "canceled":
                print(f"Batch {batch_id} canceled")

        return all_responses

    def load_saved_batches(self, batch_ids: List[str]) -> List[dict]:
        all_responses = []
        for batch in batch_ids:
            output_file_path = os.path.join(self.saver.output_path, f"{batch}.jsonl")
            with open(output_file_path, "r") as f:
                raw_responses = f.readlines()
                all_responses.extend(raw_responses)

        json_output = []
        for raw_response in all_responses:
            json_response = json.loads(raw_response)
            json_output.append(json_response)
        return json_output

    def parse_batches(self, all_responses: List[dict], output_len: int) -> List[str]:
        total_cost = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_errors = 0
        parsed_responses = []
        for res in all_responses:
            custom_id = res["custom_id"]
            if res["error"]:
                parsed_responses.append((custom_id, None))
                continue
            try:
                content = res["response"]["body"]["choices"][0]["message"].get("content", None)
                prompt_tokens = res["response"]["body"]["usage"]["prompt_tokens"]
                completion_tokens = res["response"]["body"]["usage"]["completion_tokens"]
                total_cost += (prompt_tokens * MODEL_PRICING[self.model]["prompt"]) + (
                    completion_tokens * MODEL_PRICING[self.model]["completion"]
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                parsed_responses.append((custom_id, content))
            except Exception as e:
                print(f"Error: {e}")
                parsed_responses.append((custom_id, None))
        parsed_response_list = [None] * output_len
        for parsed_response in parsed_responses:
            custom_id = int(parsed_response[0])
            parsed_response_list[custom_id] = parsed_response[1]

        # num_errors = len([res for res in parsed_responses if res[1] is None])
        num_errors = len([res for res in parsed_response_list if res is None])

        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Number of errors: {num_errors}")
        print(f"Yield: {len(parsed_response_list) / output_len * 100:.2f}%")
        return parsed_response_list

    def run_batches(self, prompts, **model_kwargs) -> List[str]:
        
        print(f"Running batch with {len(prompts)} prompts")
        file_paths = self.save_batch_files(prompts, **model_kwargs)
        print(f"Batch files saved to {file_paths}")

        print("Submitting batches")
        batch_ids = self.submit_batch(file_paths)
        print(f"Batches submitted with ids: {batch_ids}")

        print("Retrieving batches")
        all_responses = self.retrieve_batches(batch_ids)
        print(f"Retrieved {len(all_responses)} responses")

        return self.parse_batches(all_responses, len(prompts))


class OpenAIAgent(OpenAIAPI):
    def __init__(self, model: str = "gpt-4o", chunk_size: int = 32, saver: SaveTool = None):
        super().__init__(model, chunk_size, saver)

    def generate_prompts(self, system_prompts: Union[str, List], user_prompts: Union[str, List]):
            
        def input_validation(input: Union[str, List]):
            if isinstance(input, str):
                return
            if isinstance(input, list):
                for item in input:
                    if not item:
                        raise ValueError(f"Invalid input: {item} cannot be null")

        input_validation(system_prompts)
        input_validation(user_prompts)

        # verify user_prompts is list of strings
        if isinstance(user_prompts, list) and isinstance(user_prompts[0], str) and (len(user_prompts) != len(set(user_prompts))):
            print(f"Warning: User prompts are not unique. Length: {len(user_prompts)} != Unique Length: {len(set(user_prompts))}")
            print('Consider submitting unique prompts to the API to save on costs and time')

        if isinstance(system_prompts, str) and isinstance(user_prompts, str):
            system_prompts = [{"role": "system", "content": system_prompts}]
            user_prompts = [{"role": "user", "content": user_prompts}]
        elif isinstance(system_prompts, list) and isinstance(user_prompts, list):
            system_prompts = [{"role": "system", "content": prompt} for prompt in system_prompts]
            user_prompts = [{"role": "user", "content": prompt} for prompt in user_prompts]
            if len(system_prompts) != len(user_prompts):
                raise ValueError("Length of system prompts and user prompts must be equal")
        elif isinstance(system_prompts, str) and isinstance(user_prompts, list):
            system_prompts = [{"role": "system", "content": system_prompts} for _ in range(len(user_prompts))]
            user_prompts = [{"role": "user", "content": prompt} for prompt in user_prompts]
        else:
            raise ValueError(
                "Invalid input: number of system prompts must either be equal to number of user prompts or 1"
            )

        prompts = [[system_prompt, user_prompt] for system_prompt, user_prompt in zip(system_prompts, user_prompts)]
        return prompts

    def get_responses(self, prompts, batch=False, **model_kwargs):
        if batch:
            return self.run_batches(prompts, **model_kwargs)
        return asyncio.run(self.run_async(prompts, **model_kwargs))

    def format_df_prompts(self, df: pd.DataFrame, system_prompt: str, user_prompt: str):
        """
        Format the dataframe to include the system and user prompts

        Args:
            df: pandas DataFrame
            system_prompt: str, system prompt with placeholders to be formatted, placeholders should be column names in the dataframe
            user_prompt: str, user prompt with placeholders to be formatted, placeholders should be column names in the dataframe. E.g. "Add {Number 1} to {Number 2}"
        """
        df_results = df.copy()
        df_results["system_prompt"] = df.apply(lambda x: system_prompt.format(**x), axis=1)
        df_results["user_prompt"] = df.apply(lambda x: user_prompt.format(**x), axis=1)
        return df_results

    def run_df_prompts(
        self,
        df: pd.DataFrame,
        system_col="system_prompt",
        user_col="user_prompt",
        output_col="openai_response",
        batch=False,
        **model_kwargs,
    ):
        df_results = df.copy()
        system_prompts = df_results[system_col].tolist()
        user_prompts = df_results[user_col].tolist()
        prompts = self.generate_prompts(system_prompts, user_prompts)
        responses = self.get_responses(prompts, batch, **model_kwargs)
        df_results[output_col] = responses
        return df_results

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _format_user_img_prompts(self, text_prompt: str, image_path: Union[str, List]):
        text_part = {"type": "text", "text": text_prompt}
        user_prompt = [text_part]
        if isinstance(image_path, str):
            image_path = [image_path]
        for image_file in image_path:
            base64_image = self._encode_image(image_file)
            image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            user_prompt.append(image_part)
        return user_prompt

    def format_img_prompts(self, text_prompts: Union[str, List], image_paths: List):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts for _ in range(len(image_paths))]
        else:
            if len(text_prompts) != len(image_paths):
                raise ValueError("Length of text prompts and image paths must be equal")

        user_prompts = []
        for text_prompt, image_path in zip(text_prompts, image_paths):
            user_prompt = self._format_user_img_prompts(text_prompt=text_prompt, image_path=image_path)
            user_prompts.append(user_prompt)

        return user_prompts


class ChunkingAgent(OpenAIAgent):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_model: str = "gpt-4o-mini-batch",
        chunk_size: int = 32,
        iter_size: int = 8,
        system_prompt: str = None,
        user_prompt: str = "{reviews}",
        iter_size_first: int = None,
        iter_size_last: int = None,
        system_prompt_layer_first: str = None,
        system_prompt_layer_last: str = None,
        user_prompt_layer_first: str = None,
        user_prompt_layer_last: str = None,
    ):
        super().__init__(model, chunk_size)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        if "{reviews}" not in user_prompt:
            raise ValueError("user_prompt must contain the placeholder '{reviews}'")

        self.system_prompt_layer_first = system_prompt_layer_first or system_prompt
        self.user_prompt_layer_first = user_prompt_layer_first or user_prompt
        self.system_prompt_layer_last = system_prompt_layer_last or system_prompt
        self.user_prompt_layer_last = user_prompt_layer_last or user_prompt

        self.iter_size = iter_size
        self.iter_size_first = iter_size_first or iter_size
        self.iter_size_last = iter_size_last or iter_size
        self.current_model = None
        self.batch_model = batch_model

    def _gen_prompts(self, reviews: list[str], level: str):
        iter_size = self.iter_size
        user_prompt = self.user_prompt
        system_prompt = self.system_prompt
        if level == "first":
            iter_size = self.iter_size_first
            user_prompt = self.user_prompt_layer_first
            system_prompt = self.system_prompt_layer_first
        elif level == "last":
            iter_size = self.iter_size_last
            user_prompt = self.user_prompt_layer_last
            system_prompt = self.system_prompt_layer_last

        user_prompts = []
        chunks = np.array_split(reviews, math.ceil(len(reviews) / iter_size))

        for chunk in chunks:
            chunk = [f"{str(j+1)}. {text}" for j, text in enumerate(chunk)]
            review_chunk = "\n<><><>\n".join(chunk)
            user_message = user_prompt.format(reviews=review_chunk)
            user_prompts.append(user_message)

        prompts = self.generate_prompts(system_prompt, user_prompts)
        return prompts

    def _get_levels(self, reviews: list[str]):
        reviews_length = len(reviews)
        middle_length = math.ceil(math.ceil(reviews_length / self.iter_size_first) / self.iter_size_last)
        levels = math.ceil(math.log(middle_length, self.iter_size))
        if len(reviews) < self.iter_size * self.iter_size_first * self.iter_size_last:
            raise ValueError(
                f"""
                Number of reviews must be greater than {self.iter_size * self.iter_size_first * self.iter_size_last}
                Please adjust the iter sizes. 
                """
            )
        return levels + 2

    def get_summary(self, reviews: List[str], batch_trigger: int = 30000, **model_kwargs) -> str:
        new_summaries = reviews
        chunks = {}
        levels = self._get_levels(reviews)
        for i in range(0, levels):
            chunks[i] = new_summaries
            prompts = self._gen_prompts(new_summaries, "first" if i == 0 else ("last" if i == levels - 1 else "middle"))
            batch = True if len(prompts) >= batch_trigger else False
            if batch:
                self.current_model = self.batch_model
            else:
                self.current_model = self.model
            self.model = self.current_model
            new_summaries = self.get_responses(prompts, batch=batch, **model_kwargs)
            if len(new_summaries) == 1:
                if i != levels - 1:
                    print(f"Warning: Only one response received at level {i}. Last layer wasn't executed")
                break
        chunks[i + 1] = new_summaries
        self.layers = chunks

        return new_summaries[0]
