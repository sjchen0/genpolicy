import torch
import argparse
import os
from load_model import load_model, load_policy
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sampling import OrderedSampler,DiffusionSampler
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

def main(rep=1000):
    parser = argparse.ArgumentParser(description="Generate some samples")
    # parser.add_argument("--model_path", default="~/.cache/huggingface/hub/models--JingyangOu--radd-t-dce", type=str)
    parser.add_argument("--model_path", default="JingyangOu/radd-t-dce", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--method", type=str, default="tweedie") # ordered, euler, tweedie
    parser.add_argument("--strategy", type=str, default="direct") # direct, top_p, top_k, policy
    parser.add_argument("--strategy_para", type=float, default=0.8) # p for top_p, k for top_k, no use when direct 
    parser.add_argument("--perplexity", type=bool, default=True)
    parser.add_argument("--gpt_dir", type=str, default="gpt2-large")
    parser.add_argument("--perplexity_batch_size", type=int, default=1)
    parser.add_argument("--policy_path", default="example/014609/checkpoints-meta/checkpoint.pth", type=str)

    args = parser.parse_args()

    # if WORLD_SIZE not in os.environ, set it to 1
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    device = torch.device('cuda')
    model, noise = load_model(args.model_path, device)

    policy = load_policy(args.policy_path, device)

    token_dim = model.config.tokens + 1
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    order =  torch.arange(0,1024)
    if args.method == 'ordered':
        sampler = OrderedSampler(model, (args.batch_size, args.length), token_dim, args.strategy, args.strategy_para, order, device=device)
    elif args.method == 'euler' or args.method == 'tweedie':
        sampler = DiffusionSampler(args.method, model,  noise, (args.batch_size, args.length),token_dim, args.strategy, policy, args.strategy_para, device=device)
    else:
        raise ValueError(f"Method {args.method} is not valid.")
    
    ret_ppl = []
    if args.perplexity:
        eval_model = GPT2LMHeadModel.from_pretrained(args.gpt_dir).to(device).eval()
    for count in tqdm(range(rep)):

        samples = sampler.sample(args.steps)
        text_samples = tokenizer.batch_decode(samples)

        # for i in text_samples:
        #     print(i)
        #     print("=================================================")

        if args.perplexity:
            with torch.no_grad():
                # eval_model = GPT2LMHeadModel.from_pretrained(args.gpt_dir).to(device).eval()
                batches = samples.shape[0] // args.perplexity_batch_size
                total_perplexity = 0
                for j in range(batches):
                    s = samples[j * args.perplexity_batch_size:(j + 1) * args.perplexity_batch_size]
                    loss, logits = eval_model(s, labels=s)[:2]
                    logits = logits.transpose(-1, -2)
                    perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                    total_perplexity += perplexity
                total_perplexity /= batches
                # dist.all_reduce(total_perplexity)
                total_perplexity /= world_size
                # print(f"Avg Perplexity: {total_perplexity:.3f}.")
                # print("=================================================")
                ret_ppl.append(total_perplexity.item())
    
    return ret_ppl

if __name__=="__main__":
    ret_ppl = main(100)
    ret_ppl = np.array(ret_ppl)
    print(f"ppl mean: {np.mean(ret_ppl)}, ppl std: {np.std(ret_ppl)}")