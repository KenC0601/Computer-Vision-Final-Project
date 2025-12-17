from peft import LoraConfig, get_peft_model

def apply_pissa(model, r=16, lora_alpha=16, dropout=0.1, target_modules=["c_fc", "out_proj"]):

    print(f"Applying PiSSA with r={r}, alpha={lora_alpha}, dropout={dropout}, targets={target_modules}...")
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],
        init_lora_weights="pissa" # Enable PiSSA
    )
    peft_model = get_peft_model(model, config)
    return peft_model
