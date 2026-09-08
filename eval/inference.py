"""Inference backends: HuggingFace by default, vLLM if available.

Both expose: generate(prompts: list[str]) -> list[str].
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass


@dataclass
class GenKwargs:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


class Backend:
    name = "base"

    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HFBackend(Backend):
    """Transformers + generate(). Slower than vLLM but dependency-light."""
    name = "hf"

    def __init__(self, hf_repo: str, dtype: str = "bfloat16", max_model_len: int = 8192,
                 device: str = "cuda", trust_remote_code: bool = True, token: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo, trust_remote_code=trust_remote_code, padding_side="left", token=token
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_repo, dtype=torch_dtype, device_map="auto", trust_remote_code=trust_remote_code, token=token
        )
        self.model.eval()
        self.max_model_len = max_model_len
        # With device_map="auto" the model may span multiple GPUs; use the first
        # parameter's device so we move inputs to the right place in generate().
        self.device = next(self.model.parameters()).device
        self.gen_kwargs = GenKwargs()

    def _format(self, prompt: str) -> str:
        """Apply the tokenizer's chat template if it defines one."""
        if getattr(self.tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        formatted = [self._format(p) for p in prompts]
        enc = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_model_len - self.gen_kwargs.max_new_tokens,
        ).to(self.device)
        gk = self.gen_kwargs
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=gk.max_new_tokens,
                do_sample=gk.temperature > 0,
                temperature=gk.temperature if gk.temperature > 0 else 1.0,
                top_p=gk.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    def close(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(Backend):
    """vLLM backend — faster, requires `pip install vllm`."""
    name = "vllm"

    def __init__(self, hf_repo: str, dtype: str = "bfloat16", max_model_len: int = 8192,
                 device: str = "cuda", tensor_parallel_size: int = 1, token: str | None = None):
        from vllm import LLM, SamplingParams  # type: ignore

        # vLLM reads HF_TOKEN from the environment automatically; set it
        # explicitly here as well so it works even if the env var isn't set.
        if token is not None:
            os.environ.setdefault("HF_TOKEN", token)
        self.llm = LLM(
            model=hf_repo, dtype=dtype, max_model_len=max_model_len,
            trust_remote_code=True, tensor_parallel_size=tensor_parallel_size,
        )
        self.gen_kwargs = GenKwargs()
        self._SamplingParams = SamplingParams
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True, token=token)

    def _format(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, prompts: list[str]) -> list[str]:
        gk = self.gen_kwargs
        sp_kwargs = dict(
            max_tokens=gk.max_new_tokens,
            temperature=gk.temperature,
            top_p=gk.top_p,
            seed=gk.seed,
        )
        if getattr(self, "force_answer_regex", None):
            from vllm.sampling_params import StructuredOutputsParams  # type: ignore
            sp_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.force_answer_regex)
        sp = self._SamplingParams(**sp_kwargs)
        formatted = [self._format(p) for p in prompts]
        outs = self.llm.generate(formatted, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def close(self) -> None:
        import torch

        # Tear down vLLM's distributed process group so GPU memory is fully
        # released before the next model is loaded in the same process.
        try:
            from vllm.distributed.parallel_state import (  # type: ignore
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        try:
            import ray  # type: ignore
            ray.shutdown()
        except Exception:
            pass

        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLMTextBackend(Backend):
    """HuggingFace image-text-to-text models run with text-only input.
    Uses AutoProcessor + AutoModelForImageTextToText (e.g. MedGemma).

    Prompts are processed in batches with left-padding (required for causal
    decoder-only generation).  max_input_length truncates long contexts before
    they exceed the model's practical VRAM budget.
    """
    name = "vlm_hf"

    def __init__(self, hf_repo: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 1024, device: str = "cuda",
                 batch_size: int = 4, max_input_length: int = 2048,
                 token: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                        "float32": torch.float32}[dtype]
        self.processor = AutoProcessor.from_pretrained(hf_repo, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_repo, torch_dtype=torch_dtype, device_map="auto", token=token
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_input_length = max_input_length

        # Left-padding is required for batched causal LM generation.
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        results = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]

            # Tokenize each prompt independently (chat template requires
            # per-sample application), then left-pad to uniform length.
            encodings = []
            for p in batch:
                messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
                enc = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                )
                encodings.append(enc)

            if len(encodings) == 1:
                inputs = {k: v.to(self.device) for k, v in encodings[0].items()}
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = self.model.generate(**inputs,
                                              max_new_tokens=self.max_new_tokens)
                new_tok = out[0, prompt_len:]
                results.append(self.processor.decode(new_tok,
                                                     skip_special_tokens=True))
            else:
                pad_id = self.processor.tokenizer.pad_token_id
                max_len = max(e["input_ids"].shape[1] for e in encodings)
                input_ids = torch.full((len(encodings), max_len), pad_id,
                                       dtype=torch.long)
                attention_mask = torch.zeros(len(encodings), max_len,
                                             dtype=torch.long)
                for j, enc in enumerate(encodings):
                    seq_len = enc["input_ids"].shape[1]
                    input_ids[j, max_len - seq_len:] = enc["input_ids"][0]
                    attention_mask[j, max_len - seq_len:] = 1

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                    )
                for j in range(len(encodings)):
                    new_tok = out[j, max_len:]
                    results.append(self.processor.decode(new_tok,
                                                         skip_special_tokens=True))
        return results

    def close(self) -> None:
        import torch

        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFInferenceAPIBackend(Backend):
    """HuggingFace Inference Providers via the OpenAI-compatible router.

    Model IDs use the format  {hf_repo}:{provider}  e.g.
        deepseek-ai/DeepSeek-R1-Distill-Llama-8B:nscale
    Check each model's HF page (Deploy button) to find its available providers.
    """
    name = "api"

    def __init__(self, hf_repo: str, token: str, api_model: str | None = None,
                 max_new_tokens: int = 512):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
        # api_model should be "{hf_repo}:{provider}"; fall back to hf_repo if unset
        self.api_model = api_model or hf_repo
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> list[str]:
        results = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
            )
            results.append(response.choices[0].message.content)
        return results


class BedrockBackend(Backend):
    """AWS Bedrock inference via boto3 Converse API.
    AWS credentials are read from environment (AWS_ACCESS_KEY_ID etc.).
    """
    name = "bedrock"

    def __init__(self, model_id: str, region: str | None = None,
                 max_new_tokens: int = 512):
        import boto3
        region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> list[str]:
        results = []
        for prompt in prompts:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.max_new_tokens, "temperature": 0.0},
            )
            results.append(response["output"]["message"]["content"][0]["text"])
        return results


class SageMakerBackend(Backend):
    """Deploy a HuggingFace model to a SageMaker endpoint and run inference.
    Requires: pip install 'sagemaker<3.0.0' boto3
    AWS credentials are read from environment (AWS_ACCESS_KEY_ID etc.).
    The endpoint is deleted in close() to avoid ongoing charges.
    """
    name = "sagemaker"

    def __init__(self, hf_repo: str, token: str,
                 instance_type: str = "ml.g5.2xlarge",
                 num_gpus: int = 1,
                 tgi_version: str = "3.3.6",
                 region: str | None = None):
        import boto3
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

        if region:
            boto_session = boto3.Session(region_name=region)
            sagemaker_session = sagemaker.Session(boto_session=boto_session)
        else:
            sagemaker_session = None

        try:
            role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
        except ValueError:
            iam = boto3.client("iam")
            role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

        hub_env = {
            "HF_MODEL_ID": hf_repo,
            "SM_NUM_GPUS": str(num_gpus),
            "HF_TOKEN": token,
        }
        image_uri = get_huggingface_llm_image_uri("huggingface", version=tgi_version,
                                                   region_name=region or boto3.Session().region_name)
        model = HuggingFaceModel(image_uri=image_uri, env=hub_env, role=role,
                                 sagemaker_session=sagemaker_session)

        print(f"  Deploying {hf_repo} to SageMaker ({instance_type})...", flush=True)
        self.predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            container_startup_health_check_timeout=300,
        )
        print("  Endpoint ready.", flush=True)

    def generate(self, prompts: list[str]) -> list[str]:
        results = []
        for prompt in prompts:
            out = self.predictor.predict({
                "inputs": prompt,
                "parameters": {"max_new_tokens": 512},
            })
            # TGI returns either a list of dicts or a dict
            if isinstance(out, list):
                results.append(out[0].get("generated_text", str(out[0])))
            else:
                results.append(out.get("generated_text", str(out)))
        return results

    def close(self) -> None:
        print("  Deleting SageMaker endpoint...", flush=True)
        self.predictor.delete_endpoint()


def make_backend(model_cfg: dict, backend: str = "auto", device: str = "cuda",
                 token: str | None = None) -> Backend:
    # Allow per-model backend override (e.g. "sagemaker" for VLMs)
    effective_backend = model_cfg.get("backend", backend)

    if effective_backend == "api":
        if not token:
            raise ValueError("--hf-token / HF_TOKEN is required for the api backend")
        return HFInferenceAPIBackend(
            hf_repo=model_cfg["hf_repo"],
            token=token,
            api_model=model_cfg.get("api_model"),
            max_new_tokens=model_cfg.get("max_new_tokens", 512),
        )
    if effective_backend == "bedrock":
        return BedrockBackend(
            model_id=model_cfg["bedrock_model_id"],
            region=os.environ.get("AWS_DEFAULT_REGION"),
            max_new_tokens=model_cfg.get("max_new_tokens", 512),
        )
    if effective_backend == "vlm_hf":
        return VLMTextBackend(
            hf_repo=model_cfg["hf_repo"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_new_tokens=model_cfg.get("max_new_tokens", 128),
            batch_size=model_cfg.get("vlm_batch_size", 4),
            max_input_length=model_cfg.get("vlm_max_input_length", 2048),
            device=device,
            token=token,
        )
    if effective_backend == "sagemaker":
        if not token:
            raise ValueError("--hf-token / HF_TOKEN is required for the sagemaker backend")
        return SageMakerBackend(
            hf_repo=model_cfg["hf_repo"],
            token=token,
            instance_type=model_cfg.get("sm_instance_type", "ml.g5.2xlarge"),
            num_gpus=model_cfg.get("sm_num_gpus", 1),
            region=os.environ.get("AWS_DEFAULT_REGION"),
        )
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except Exception:
            backend = "hf"
    if backend == "vllm":
        return VLLMBackend(
            hf_repo=model_cfg["hf_repo"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_model_len=model_cfg.get("max_model_len", 8192),
            device=device,
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
            token=token,
        )
    return HFBackend(
        hf_repo=model_cfg["hf_repo"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=model_cfg.get("max_model_len", 8192),
        device=device,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        token=token,
    )


def timed_generate(backend: Backend, prompts: list[str]) -> tuple[list[str], list[int]]:
    """Returns (responses, per_prompt_wall_ms_approx). Wall time is split evenly
    across a batch — good enough for logging, not profiling."""
    t0 = time.monotonic()
    outs = backend.generate(prompts)
    total_ms = int((time.monotonic() - t0) * 1000)
    per = total_ms // max(len(prompts), 1)
    return outs, [per] * len(outs)
