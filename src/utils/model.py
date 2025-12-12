import transformers
import torch
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
from typing import Sequence, Union
from src.utils.fabric import FabricPlus

from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


def setup_fabric(
    precision: str,
    devices: Union[str, int, Sequence[int]] = "auto",
    strategy: str = "auto",
):
    fabric = FabricPlus(precision=precision, devices=devices, strategy=strategy)
    fabric.launch()
    return fabric


def load_model_and_tokenizer(model_name: str, fabric: FabricPlus):
    use_mistral_backend = "Ministral-3" in model_name and MistralCommonBackend is not None
    
    if use_mistral_backend:
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with fabric.init_module():
        if use_mistral_backend:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            
        for p in model.parameters():
            p.requires_grad_(False)
    
    if model.config.bos_token_id is None and hasattr(tokenizer, "bos_token_id"):
        model.config.bos_token_id = tokenizer.bos_token_id

    model = fabric.to_device(model)
    model.eval()
    model = torch.compile(model)
    return model, tokenizer
