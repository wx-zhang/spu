
from transformers import  CLIPModel
from peft import LoraConfig, get_peft_model


def prepare_hf_lora_model(args, peft=True):

    if args.model == 'ViT-B/16':
        checkpoint = "openai/clip-vit-base-patch16"
    else:
        raise NotImplementedError

    model = CLIPModel.from_pretrained(checkpoint)

    if not peft:
        return model.cuda()

    config = LoraConfig(
        target_modules=['v_proj', 'q_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.1
    )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)
    print (model)
    return lora_model.cuda()

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
