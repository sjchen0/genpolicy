import abc
import torch
import torch.nn.functional as F
from catsample import sample_with_strategy, sample_categorical


import abc


class Sampler(abc.ABC):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, device=torch.device('cuda')):
        super().__init__()
        self.model = model
        self.batch_dims = batch_dims
        self.device = device
        self.strategy = strategy
        self.strategy_para = strategy_para
        self.token_dim = token_dim

    @abc.abstractmethod
    def sample(self, steps):
        raise NotImplementedError


class DiffusionSampler(Sampler):
    def __init__(self, method, model, noise, batch_dims, token_dim, strategy, policy=None, strategy_para=None, eps=1e-5, device=torch.device('cuda')):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.noise = noise
        self.eps = eps
        self.method = method
        self.update_cnt = 0
        self.policy = policy

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        if self.strategy == 'direct':
            return self.direct_sample(steps, proj_fun)
        elif self.strategy == 'policy':
            return self.policy_sample(steps, proj_fun)
        elif self.strategy == 'confidence':
            return self.confidence_sample(steps, proj_fun)
        else:
            return self.strateged_sample(steps, proj_fun)

    @torch.no_grad()
    def strateged_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        logits = torch.zeros(*self.batch_dims, self.token_dim, dtype=self.model.dtype).to(self.device)

        for i in range(steps):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps)
            if changed.any():
                logits[changed] = self.model.logits(x[changed])
                self.update_cnt += changed.sum().item()
            mask = x == self.token_dim - 1
            update_indices = (mask & (torch.rand(*self.batch_dims).to(self.device) < update_rate)) if i < steps - 1 else mask
            update_logits = logits[update_indices]
            update_x = sample_with_strategy(update_logits, self.strategy, self.strategy_para)
            x_old = x.clone()
            x[update_indices] = update_x
            changed = (x != x_old).any(dim=-1)
        return x

    @torch.no_grad()
    def policy_sample(self, steps, proj_fun=lambda x: x):
        '''
        Use deterministic policy to sample
        '''
        self.model.eval()
        self.policy.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        # p_condition = torch.zeros(*self.batch_dims, self.token_dim, dtype=torch.float16).to(self.device)
        for i in range(steps):
            t = timesteps[i]
            batch_t = t * torch.ones(self.batch_dims[0], device=self.device)
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3
            # if changed.any():
            mask = x == self.token_dim - 1
            # print("mask cnt", mask.sum(-1))
            log_condition, hidden_state = self.model.forward_with_hidden(x) # (B, L, V+1), (B, L, h)
            policy = self.policy(hidden_state, log_condition, batch_t)[:,:,1] # (B, L)
            # policy = policy + torch.randn_like(policy) * 2e-3
            # print(policy.min().item(), policy.max().item())
            vocab_probs = torch.ones_like(log_condition, dtype=self.policy.dtype)
            vocab_probs[:,:,:-1] = F.softmax(log_condition[:, :, :-1], dim=-1) # (B, L, V)

            # decide unmask set by deterministic top k
            # policy[~mask] = -1.
            # _, indices = torch.topk(policy, policy.shape[1] // steps, dim=-1)
            # unmask_set = torch.zeros_like(mask, dtype=torch.bool)
            # unmask_set.scatter_(1, indices, True)

            # decide unmask set by sampling
            policy[~mask] = 0
            # keep top-p portion

            if mask.sum(-1).min().item() > policy.shape[1] // steps:
                top_p = 0.99
                policy = policy / policy.sum(dim=-1, keepdim=True)
                sorted_probs, sorted_idx = policy.sort(dim=-1, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                cutoff = cumulative <= top_p
                cutoff[:, 0] = True
                filtered = torch.zeros_like(policy)
                filtered.scatter_(1, sorted_idx[cutoff].view(policy.size(0), -1), 
                                    sorted_probs[cutoff].view(policy.size(0), -1))
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                policy = filtered

            unmask_set = torch.zeros_like(mask, dtype=torch.bool)
            unmask_indices = torch.zeros((policy.shape[0], policy.shape[1] // steps), dtype=torch.int64, device=policy.device)
            for b in range(policy.shape[0]):
                idx = torch.multinomial(policy[b], policy.shape[1] // steps, replacement=False)
                # print("sampled indices len:", len(idx), idx)
                unmask_indices[b] = idx
            unmask_set.scatter_(1, unmask_indices, True)
            

            p_condition = vocab_probs[:,:,:-1]
            p_condition_unmask = p_condition[unmask_set]

            update_x_unmask = sample_categorical(p_condition_unmask.to(torch.float64))
            x_old = x.clone()
            # import ipdb; ipdb.set_trace()
            x[unmask_set] = update_x_unmask
            changed = (x != x_old).any(dim=-1)

            # print("num unmask", (x != x_old).sum(-1), (x != x_old).nonzero(as_tuple=True)[1])
            self.update_cnt += changed.sum().item()
        mask = x == self.token_dim - 1
        # print("total unmasked:", mask.sum())
        # print("returning", x.shape)
        return x

    @torch.no_grad()
    def direct_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        p_condition = torch.zeros(*self.batch_dims, self.token_dim, dtype=torch.float32).to(self.device)
        for i in range(steps):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3
            if changed.any():
                mask = x == self.token_dim - 1
                p_condition[changed] = self.model(x[changed]).exp()
                p_condition_mask = p_condition[mask]
            probs_mask = p_condition_mask * update_rate
            probs_mask[..., -1] = 1 - update_rate
            update_x_mask = sample_categorical(probs_mask.to(torch.float64))
            x_old = x.clone()
            x[mask] = update_x_mask
            changed = (x != x_old).any(dim=-1)
            # print("num unmask", (x != x_old).sum(-1), (x != x_old).nonzero(as_tuple=True)[1])
            self.update_cnt += changed.sum().item()
        return x
    
    @torch.no_grad()
    def confidence_sample(self, steps, proj_fun=lambda x: x):
        '''
        Use deterministic policy to sample
        '''
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)

        x = proj_fun(x)
        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        changed = torch.ones(self.batch_dims[0], dtype=torch.bool)
        # p_condition = torch.zeros(*self.batch_dims, self.token_dim, dtype=torch.float16).to(self.device)
        for i in range(steps):
            t = timesteps[i]
            batch_t = t * torch.ones(self.batch_dims[0], device=self.device)
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3
            # if changed.any():
            mask = x == self.token_dim - 1
            log_condition, hidden_state = self.model.forward_with_hidden(x) # (B, L, V+1), (B, L, h)
            # print(log_condition.dtype)
            
            # confidence ver 1
            # policy = torch.amax(log_condition[:,:,:-1], dim=-1) # (B, L)

            # confidence ver 2
            v, idx = torch.topk(log_condition[:,:,:-1], 2, dim=-1)
            policy = (v[:,:,0] - v[:,:,1]).abs()
            policy += torch.randn_like(policy) * 1e-0

            # eps = 1e-5 * torch.randint(0, 2, policy.shape, device=policy.device).float()
            # policy = policy + eps
            # print(policy.min().item(), policy.max().item())
            vocab_probs = torch.ones_like(log_condition, dtype=self.policy.dtype)
            vocab_probs[:,:,:-1] = F.softmax(log_condition[:, :, :-1], dim=-1) # (B, L, V)

            # sample unmask set by Bernoulli trials --- seems not to work well
            # alternative: unmask the top-k tokens with highest policy value

            policy[~mask] = -1e4
            values, indices = torch.topk(policy, policy.shape[1] // steps, dim=-1)
            # print(values)
            unmask_set = torch.zeros_like(mask, dtype=torch.bool)
            unmask_set.scatter_(1, indices, True)

            p_condition = log_condition.exp()
            p_condition = p_condition[:,:,:-1] / p_condition[:,:,:-1].sum(-1, keepdim=True)
            p_condition_unmask = p_condition[unmask_set]

            update_x_unmask = sample_categorical(p_condition_unmask.to(torch.float64))
            x_old = x.clone()
            # import ipdb; ipdb.set_trace()
            x[unmask_set] = update_x_unmask
            changed = (x != x_old).any(dim=-1)

            # print("num unmask", (x != x_old).sum(-1), (x != x_old).nonzero(as_tuple=True)[1])
            self.update_cnt += changed.sum().item()
        mask = x == self.token_dim - 1
        return x

    def get_update_rate(self, t, steps):
        dt = (1 - self.eps) / steps
        curr_sigma, next_sigma = self.noise(t)[0], self.noise(t - dt)[0]
        d_curr_sigma = self.noise(t)[1]
        if self.method == 'tweedie':
            update_rate = ((-next_sigma).exp() - (-curr_sigma).exp()) / (1 - (-curr_sigma).exp())
        elif self.method == 'euler':
            update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
        return update_rate


class OrderedSampler(Sampler):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, order=None, device=torch.device('cuda')):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.order = order

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        order = torch.randperm(1024) if self.order is None else self.order
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)
        x = proj_fun(x)

        for i in range(steps):
            logits = self.model.logits(x)
            update_logits = logits[:, order[i], :-1]
            x[:, order[i]] = sample_with_strategy(update_logits, self.strategy, self.strategy_para)
        return x
