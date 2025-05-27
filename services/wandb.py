import torch
import wandb

def get_tags(args):
    tags = [args.equation, args.method]
    if args.errorbounds:
        tags.append("eb")
    if args.bundle:
        tags.append("bundle")
    if args.inverse:
        tags.append("inverse")
    if args.output_variance:
        tags.append("ov")
    return tags

def init_wandb_run(equation, method_config, method_config_bundle, inverse_config, experiment_name, args):
    wandb.init(project="thesis", name=experiment_name, tags=get_tags(args),
    config={
        "equation": equation.__dict__,
        "method": method_config_bundle.__dict__ if args.bundle else method_config.__dict__,
        "inverse": inverse_config.__dict__ if args.inverse else "None",
        "inverse_dataset": args.inverse_dataset if args.inverse else "None",
        "equation": args.equation,
        "method": args.method,
        "variance": "bounds" if args.errorbounds else ("ov" if args.output_variance else "homo"),
        "scheme": "bundle" if args.bundle else "forward",
        "inverse": args.inverse,
        "device": "gpu" if args.gpu else "cpu",
        "torch_device": torch.cuda.is_available(),
        "residual_loss": args.res_loss,
        })