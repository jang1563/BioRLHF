"""
Model utilities for BioRLHF.

This module provides helper functions for loading models, configuring
quantization, and setting up LoRA adapters.
"""

from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytes quantization configuration.

    Args:
        load_in_4bit: Use 4-bit quantization.
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4').
        bnb_4bit_compute_dtype: Compute dtype for quantized operations.
        bnb_4bit_use_double_quant: Use nested quantization.

    Returns:
        BitsAndBytesConfig for model loading.
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA alpha (scaling factor).
        target_modules: Modules to apply LoRA to.
        lora_dropout: Dropout probability for LoRA layers.
        bias: Bias training strategy ('none', 'all', or 'lora_only').
        task_type: Task type for the model.

    Returns:
        LoraConfig for PEFT.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def load_model_for_inference(
    model_path: str,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    use_4bit: bool = True,
    device_map: str = "auto",
    merge_adapters: bool = False,
) -> tuple:
    """
    Load a fine-tuned model for inference.

    Args:
        model_path: Path to the fine-tuned model/adapters.
        base_model: Base model name (for adapter loading).
        use_4bit: Use 4-bit quantization.
        device_map: Device mapping strategy.
        merge_adapters: Merge LoRA adapters into base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Quantization config
    bnb_config = get_quantization_config() if use_4bit else None

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load adapters
    model = PeftModel.from_pretrained(model, model_path)

    if merge_adapters:
        model = model.merge_and_unload()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Use sampling (vs greedy decoding).

    Returns:
        Generated response text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()
