import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from noise_lib import add_noise_t, add_noise_lambda, add_noise_k, add_noise_trajectory

def Batch_Uniform_Sampler(B, type = 'naive', device = 'cuda'):
    def vdm_sampler(B, device):
        u_0 = torch.rand(1, device=device)  # Sample u_0 from U(0, 1)
        t = [(u_0 + i / B) % 1 for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    
    def decoupled_sampler(B, device):
        u = torch.rand(B, device=device)  # Sample B independent values from U(0, 1)
        t = [(u[i] + i) / B for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    if type == 'naive':
        return torch.rand(B, device = device)
    elif type == 'vdm':
        return vdm_sampler(B, device)
    elif type == 'decoupled':
        return decoupled_sampler(B, device)
    else:
        raise ValueError(f"{type} not valid")
    

def get_loss_fn(noise, token_dim, train, sampling_eps=1e-3, loss_type='lambda_DCE',order = torch.arange(1024)):
    def t_DSE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score (reuse log_condition to save memory)
        log_condition -=esigm1.log()[...,None]

        scaling_factor = 1 / esigm1.expand_as(perturbed_batch)
        
        # compute three terms
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        # add negative term
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        # add pos term
        loss[masked_index] += log_condition[masked_index][:, :-1].exp().sum(dim=-1)

        # add const term 
        loss[masked_index] += scaling_factor[masked_index] * (scaling_factor[masked_index].log() - 1)
        return (dsigma * loss).sum(dim=-1)

    def t_DCE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score 
        log_condition -=esigm1.log()[...,None]

        # compute DCE loss
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        return (dsigma * loss).sum(dim=-1)

    def lambda_DCE_loss(model, batch, cond = None):
        # sample lambda and add noise
        # Lambda = torch.rand(batch.shape[0], device=batch.device)
        Lambda = Batch_Uniform_Sampler(batch.shape[0], type = 'decoupled', device = batch.device)
        perturbed_batch = add_noise_lambda(batch, Lambda, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]
        
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1).to(torch.float64)/Lambda.to(torch.float64)
        return loss

    def k_DCE_loss(model, batch, cond = None): # any-order ar loss
        # sample k and add noise
        k = torch.randint(1, batch.shape[1] + 1 ,(batch.shape[0],),device=batch.device)
        perturbed_batch = add_noise_k(batch, k, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1)/k * batch.shape[1]
        return loss.to(torch.float32)

    if loss_type =='ar_forward':
        order = torch.arange(0,1024)
    elif loss_type =='ar_backward':
        order = torch.arange(1023,-1,-1)
    else:
        order = torch.arange(1024)

    def ar_loss(model, batch):
        nonlocal order
        if loss_type == 'ar_random':
            order = torch.randperm(1024)
        if train:
            model.train()
        else:
            model.eval()
        loss = 0
        for i in range(batch.shape[1]):
            masked_batch = batch.clone()
            masked_batch[:,order[i:]] = token_dim - 1
            p_log_condition_i = model(masked_batch)[:,order[i]]
            loss += - p_log_condition_i[torch.arange(batch.shape[0]),batch[:,order[i]]].to(torch.float32)
        return loss
    
    if loss_type == 'ar_forward' or loss_type == 'ar_backward' or loss_type == 'ar_random': # ar loss for a fix order
        return ar_loss
    elif loss_type =='lambda_DCE':
        return lambda_DCE_loss
    elif loss_type =='t_DCE':
        return t_DCE_loss
    elif loss_type =='t_DSE':
        return t_DSE_loss
    elif loss_type =='k_DCE':  # any-order ar loss
        return k_DCE_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported yet!')


def get_policy_loss_fn(noise, token_dim, train, discrete_timesteps, num_trajectories=2, sampling_eps=1e-3, loss_type='lambda_DCE',order = torch.arange(1024)):
    def policy_log_loss_uniform(score_model, policy_model, batch, cond = None):
        # 1. given a batch, sample multiple trajectories at discrete timesteps 0 < t1 < ... < tK < 1
        # 2. at each state of each trajectory, looking at the unmasked places,
        #    evaluate (log_policy_model(x_k) + log_score_model(x_k-1 | x_k)).sum().exp()
        # 3. then aggregate and take negative log
        score_model.eval()
        if train:
            policy_model.train()
        else:
            policy_model.eval()

        total_loss = torch.zeros(batch.shape[0], device=batch.device)
        for n in range(num_trajectories):
            #print(n)
            batch_discrete_timesteps = discrete_timesteps.unsqueeze(0).repeat(batch.shape[0], 1) # (B, K)
            sigma, dsigma = noise(batch_discrete_timesteps) # K many values, one at each timestep
            perturbed_batch_trajectory = add_noise_trajectory(batch, sigma, token_dim - 1) # (B, K, L)
            log_trajectory_metric = torch.zeros(batch.shape[0], device=batch.device)
            for k_full in range(2):
                k = torch.randint(0, perturbed_batch_trajectory.shape[1], (1,)).item()
                batch_k = perturbed_batch_trajectory[:,k,:]
                batch_km1 = perturbed_batch_trajectory[:,k-1,:] if k > 0 else batch
                batch_t = batch_discrete_timesteps[:,k]
                with torch.no_grad():
                    log_condition, hidden_state = score_model.forward_with_hidden(batch_k) # (B, L, V+1), (B, L, h)
                #log_condition = log_condition.to(batch.device)
                #hidden_state = hidden_state.to(batch.device)
                vocab_probs = torch.ones_like(log_condition, dtype=policy_model.module.dtype)
                vocab_probs[:,:,:-1] = F.softmax(log_condition[:, :, :-1], dim=-1) # (B, L, V)
                unmasked = (batch_k != batch_km1).to(vocab_probs.dtype)
                # import ipdb; ipdb.set_trace()
                target_onehot = F.one_hot(batch_km1, num_classes=vocab_probs.shape[-1])
                vocab_probs = (vocab_probs * target_onehot).sum(-1) # (B, L)
                policy = policy_model(hidden_state, log_condition, batch_t).squeeze(-1) # (B, L)
                # print(k, vocab_probs.max(), vocab_probs.min(), policy.max(), policy.min())
                
                log_step_metric = (((vocab_probs + 1e-20).log() + (policy + 1e-20).log()) * unmasked).mean(-1) # (B,)
                # print(n, k_full, vocab_probs.sum(-1))
                # step_metric = (vocab_probs * policy * unmasked).prod(-1)

                log_trajectory_metric += log_step_metric

                # import ipdb; ipdb.set_trace()
            loss = -log_trajectory_metric
            total_loss += loss
        return total_loss

    def policy_log_loss_bidirection(score_model, policy_model, batch, cond = None):
        # 1. given a batch, sample multiple trajectories at discrete timesteps 0 < t1 < ... < tK < 1
        # 2. at each state of each trajectory, looking at the unmasked places,
        #    evaluate (log_policy_model(x_k) + log_score_model(x_k-1 | x_k)).sum().exp()
        # 3. then aggregate and take negative log
        score_model.eval()
        if train:
            policy_model.train()
        else:
            policy_model.eval()

        total_loss = torch.zeros(batch.shape[0], device=batch.device)
        B, L = batch.shape
        K = discrete_timesteps.shape[0]
        num_unmask = L // K

        for n in range(num_trajectories):
            # sample discrete timesteps
            batch_km1 = batch.clone()
            for k in range(K):
                with torch.no_grad():
                    log_condition, hidden_state = score_model.forward_with_hidden(batch_km1)
                batch_t = discrete_timesteps[k] * torch.ones(B, device=batch.device)
                forward_policy = policy_model(hidden_state, log_condition, batch_t)[:,:,0] # (B, L)
                mask = batch_km1 == token_dim - 1
                forward_policy[mask] = 0.
                forward_set = torch.zeros_like(batch, dtype=torch.bool)
                forward_indices = torch.zeros((forward_policy.shape[0], num_unmask), dtype=torch.int64, device=forward_policy.device)
                for b in range(forward_policy.shape[0]):
                    idx = torch.multinomial(forward_policy[b], num_unmask, replacement=False)
                    # print("sampled indices len:", len(idx), idx)
                    forward_indices[b] = idx
                forward_set.scatter_(1, forward_indices, True)
                batch_k = batch_km1.clone()
                batch_k[forward_set] = token_dim - 1

                # forward_num_mask = torch.randint(1, L - num_unmask, (1,)).item()
                # batch_forward_ratio = forward_num_mask / L * torch.ones(B, device=batch.device, dtype=policy_model.module.dtype)
                
                # _, idx_km1 = torch.topk(forward_policy, forward_num_mask, dim=-1)
                # batch_km1 = batch.clone()
                # mask_km1 = torch.zeros_like(batch_km1, dtype=torch.bool)
                # mask_km1.scatter_(1, idx_km1, True)
                # batch_km1[mask_km1] = token_dim - 1
                
                # _, idx_k = torch.topk(forward_policy, forward_num_mask + num_unmask, dim=-1)
                # batch_k = batch.clone()
                # mask_k = torch.zeros_like(batch_k, dtype=torch.bool)
                # mask_k.scatter_(1, idx_k, True)
                # batch_k[mask_k] = token_dim - 1

                with torch.no_grad():
                    log_condition, hidden_state = score_model.forward_with_hidden(batch_k)
                
                vocab_probs = torch.ones_like(log_condition, dtype=policy_model.module.dtype)
                vocab_probs[:,:,:-1] = F.softmax(log_condition[:,:,:-1], dim=-1) # (B, L, V)
                unmasked = (batch_k != batch_km1).to(vocab_probs.dtype)
                target_onehot = F.one_hot(batch_km1, num_classes=vocab_probs.shape[-1])
                vocab_probs = (vocab_probs * target_onehot).sum(-1) # (B, L)
                backward_policy = policy_model(hidden_state, log_condition, batch_t)[:,:,1]

                log_step_metric = (((vocab_probs + 1e-20).log() + (backward_policy + 1e-20).log()) * unmasked).mean(-1)
                total_loss -= log_step_metric

                batch_km1 = batch_k.clone()

        return total_loss
    
    def policy_log_loss_debug(score_model, policy_model, batch, cond = None):
        # this is for debugging and analysis purpose only
        score_model.eval()
        policy_model.eval()

        total_loss = torch.zeros(batch.shape[0], device=batch.device)
        B, L = batch.shape
        for n in range(num_trajectories):
            #print(n)
            batch_discrete_timesteps = discrete_timesteps.unsqueeze(0).repeat(batch.shape[0], 1) # (B, K)
            sigma, dsigma = noise(batch_discrete_timesteps) # K many values, one at each timestep
            perturbed_batch_trajectory = add_noise_trajectory(batch, sigma, token_dim - 1) # (B, K, L)
            log_trajectory_metric = torch.zeros(batch.shape[0], device=batch.device)
            for k_full in range(1):
                k = 14
                # k = torch.randint(0, perturbed_batch_trajectory.shape[1], (1,)).item()
                # batch_k = perturbed_batch_trajectory[:,k,:]
                batch_km1 = perturbed_batch_trajectory[:,k-1,:] if k > 0 else batch

                signals = []

                for i in range(L):
                    batch_k = batch_km1.clone()
                    batch_k[:,i] = token_dim - 1

                    batch_t = batch_discrete_timesteps[:,k]
                    with torch.no_grad():
                        log_condition, hidden_state = score_model.forward_with_hidden(batch_k) # (B, L, V+1), (B, L, h)
                    #log_condition = log_condition.to(batch.device)
                    #hidden_state = hidden_state.to(batch.device)
                    vocab_probs = torch.ones_like(log_condition, dtype=policy_model.module.dtype)
                    vocab_probs[:,:,:-1] = F.softmax(log_condition[:, :, :-1], dim=-1) # (B, L, V)
                    unmasked = (batch_k != batch_km1).to(vocab_probs.dtype)
                    # import ipdb; ipdb.set_trace()
                    target_onehot = F.one_hot(batch_km1, num_classes=vocab_probs.shape[-1])
                    vocab_probs = (vocab_probs * target_onehot).sum(-1) # (B, L)
                    with torch.no_grad():
                        policy = policy_model(hidden_state, log_condition, batch_t).squeeze(-1) # (B, L)
                    # print(k, vocab_probs.max(), vocab_probs.min(), policy.max(), policy.min())
                    
                    log_step_metric = (((vocab_probs + 1e-20).log() + (policy + 1e-20).log()) * unmasked).mean(-1) # (B,)
                    print(n, i)
                    signals.append(vocab_probs.sum(-1)[0])
                    # step_metric = (vocab_probs * policy * unmasked).prod(-1)

                    log_trajectory_metric += log_step_metric

                # import ipdb; ipdb.set_trace()
                print(signals)
                
            loss = -log_trajectory_metric
            total_loss += loss
        return total_loss
    
    if loss_type == 'policy_log_loss':
        return policy_log_loss_bidirection
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported yet!')


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, token_dim,  train, optimize_fn, accum, loss_type):
    loss_fn = get_loss_fn(noise, token_dim, train, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


def get_policy_step_fn(noise, token_dim, train, discrete_timesteps, optimize_fn, accum, loss_type):
    policy_loss_fn = get_policy_loss_fn(noise, token_dim, train, discrete_timesteps, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        score_model = state['score_model']
        policy_model = state['policy_model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = policy_loss_fn(score_model, policy_model, batch, cond=cond).mean() / accum
            
            # print(loss)
            scaler.scale(loss).backward()

            # check gradient
            # for name, param in policy_model.named_parameters():
            #     print(name, param.requires_grad)
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean().item())
            #     else:
            #         print(name, "has no grad")


            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, policy_model.parameters(), step=state['step'])
                state['ema'].update(policy_model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(policy_model.parameters())
                ema.copy_to(policy_model.parameters())
                loss = policy_loss_fn(score_model, policy_model, batch, cond=cond).mean()
                ema.restore(policy_model.parameters())

        return loss

    return step_fn
