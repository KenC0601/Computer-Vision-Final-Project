
from peft import LoraConfig, get_peft_model

def apply_lora(model, r=16, lora_alpha=16, target_modules=["c_fc", "out_proj"]):
    """
    Applies standard LoRA to the model.
    """
    print(f"Applying LoRA with r={r}, alpha={lora_alpha}, targets={target_modules}...")
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"], # We need to train the classifier head too
    )
    peft_model = get_peft_model(model, config)
    return peft_model
