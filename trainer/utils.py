import wandb
import os
import warnings
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def logging(x_name, x_value, y_name, y_value, args):
    if args.wandb:
        wandb.define_metric(x_name)
        wandb.define_metric(y_name, step_metric=x_name)
        wandb.log({
            x_name: x_value,
            y_name: y_value
        })
        
def get_ckpt_save_path(args, task, end='pt'):
    CHECKPOINT_NAME = f"task{task}.{end}"

    path = os.path.join(args.save_base_path, args.name,
                        "checkpoints_" + CHECKPOINT_NAME)
    return path        

def resume(args, task, model, optimizer=None, scaler=None):
    start_epoch = 0

    resume_from = get_ckpt_save_path(args, task)
    if not os.path.exists(resume_from):
        warnings.warn('Warning: No ckpt to resume')
        return model, optimizer, scaler, start_epoch


    checkpoint = torch.load(resume_from, map_location='cpu')
    if 'epoch' in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        model.load_state_dict(sd)
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)

    return model, optimizer, scaler, start_epoch