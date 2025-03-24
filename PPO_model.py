import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from doagnn import RGCN, GraphSAGE
from mlp import MLPCritic, MLPActor
import torch.optim.lr_scheduler as lr_scheduler

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []

        self.ope_step_batch = []
        self.end_ope_biases_batch = []
        self.opes_appertain_batch = []

    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]

        del self.ope_step_batch[:]
        del self.end_ope_biases_batch[:]
        del self.opes_appertain_batch[:]

    def print_memories(self):
        print("states :{}".format(self.states))
        print("logprobs :{}".format(self.logprobs))
        print("rewards :{}".format(self.rewards))
        print("is_terminals :{}".format(self.is_terminals))
        print("action_indexes :{}".format(self.action_indexes))
        print("ope_ma_adj :{}".format(self.ope_ma_adj))
        print("ope_pre_adj :{}".format(self.ope_pre_adj))
        print("ope_sub_adj :{}".format(self.ope_sub_adj))
        print("batch_idxes :{}".format(self.batch_idxes))
        print("raw_opes :{}".format(self.raw_opes))
        print("raw_mas :{}".format(self.raw_mas))
        print("proc_time :{}".format(self.proc_time))
        print("jobs_gather :{}".format(self.jobs_gather))
        print("eligible :{}".format(self.eligible))
        print("nums_opes :{}".format(self.nums_opes))


class Scheduler(nn.Module):
    def __init__(self, model_paras, num_types=3):
        super(Scheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]
        self.out_size_ma = model_paras["out_size_ma"]
        self.in_size_ope = model_paras["in_size_ope"]
        self.out_size_ope = model_paras["out_size_ope"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]
        self.actor_dim = model_paras["actor_in_dim"]
        self.critic_dim = model_paras["critic_in_dim"]
        self.n_latent_actor = model_paras["n_latent_actor"]
        self.n_latent_critic = model_paras["n_latent_critic"]
        self.n_hidden_actor = model_paras["n_hidden_actor"]
        self.n_hidden_critic = model_paras["n_hidden_critic"]
        self.action_dim = model_paras["action_dim"]
        self.discount_r = model_paras["discount_r"]
        self.agg_func = model_paras["agg_func"]
        self.num_bases = model_paras["num_bases"]
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        self.num_types = num_types

        self.get_fir_operations = nn.ModuleList()

        # R-GCN
        self.get_fir_operations.append(RGCN((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                         self.dropout, self.dropout, activation=F.elu, num_types=self.num_types, num_bases=self.num_bases))

        for i in range(1,len(self.num_heads)):
            self.get_fir_operations.append(RGCN((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                         self.dropout, self.dropout, activation=F.elu, num_types=self.num_types, num_bases=self.num_bases))

        self.get_operations = nn.ModuleList()
        self.get_operations.append(GraphSAGE([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(GraphSAGE([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))


        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        batch_size = batch_idxes.size(0)

        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = self.feature_normalize(proc_time)
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        batch_idxes = state.batch_idxes
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        for i in range(len(self.num_heads)):

            h_fir_ope = self.get_fir_operations[i](state.ope_ma_adj_batch, state.batch_idxes, features)
            features = (h_fir_ope, features[1], features[2])

            h_opes = self.get_operations[i](state.ope_step_batch, state.end_ope_biases_batch,
                                            state.opes_appertain_batch, self.discount_r, state.batch_idxes, self.agg_func, features)

        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)
        else:
            h_opes_pooled = h_opes.mean(dim=-2)

        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)

        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                                                                   ope_step_batch[..., :, None].expand(-1, -1,state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])

        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])

        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return

        h_actions = torch.cat((h_jobs_padding, h_opes_pooled_padding), dim=-1).transpose(1, 2)
        h_pooled = h_opes_pooled
        mask = eligible.transpose(1, 2).flatten(1)


        scores = self.actor(h_actions).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

            memories.ope_step_batch.append(copy.deepcopy(state.ope_step_batch))
            memories.end_ope_biases_batch.append(copy.deepcopy(state.end_ope_biases_batch))
            memories.opes_appertain_batch.append(copy.deepcopy(state.opes_appertain_batch))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):

        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()

        else:
            action_indexes = action_probs.argmax(dim=1)

        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        if flag_train == True:

            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, ope_step_batch, end_ope_biases_batch, opes_appertain_batch, flag_sample=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        for i in range(len(self.num_heads)):

            h_fir_ope = self.get_fir_operations[i](ope_ma_adj, batch_idxes, features)
            features = (h_fir_ope, features[1], features[2])

            h_opes = self.get_operations[i](ope_step_batch, end_ope_biases_batch,
                                            opes_appertain_batch, self.discount_r, batch_idxes, self.agg_func, features)
            features = (h_opes, features[1], features[2])
            features = (h_opes, features[1], features[2])

        h_opes_pooled = h_opes.mean(dim=-2)
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_actions = torch.cat((h_jobs_padding, h_opes_pooled_padding), dim=-1).transpose(1, 2)
        h_pooled = h_opes_pooled

        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None, num_types=3):
        self.lr = train_paras["lr"]
        self.betas = train_paras["betas"]
        self.gamma = train_paras["gamma"]
        self.eps_clip = train_paras["eps_clip"]
        self.K_epochs = train_paras["K_epochs"]
        self.A_coeff = train_paras["A_coeff"]
        self.vf_coeff = train_paras["vf_coeff"]
        self.entropy_coeff = train_paras["entropy_coeff"]
        self.num_envs = num_envs
        self.device = model_paras["device"]
        self.policy = Scheduler(model_paras, num_types).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=3, verbose=True)

        self.gae_lambda = train_paras["GAE_lambda"]


    def find_parameter(self, module_list, module_type, parameter_name):
        for module in module_list:
            if isinstance(module, module_type):
                for name, param in module.named_parameters():
                    if name == parameter_name:
                        return param
            elif isinstance(module, nn.ModuleList) or isinstance(module, nn.ModuleDict):
                found_param = self.find_parameter(module, module_type, parameter_name)
                if found_param is not None:
                    return found_param
        return None


    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0,1).flatten(0,1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)

        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0,1).flatten(0, 1)

        old_ope_step_batch = torch.stack(memory.ope_step_batch, dim=0).transpose(0, 1).flatten(0, 1)
        old_end_ope_biases_batch = torch.stack(memory.end_ope_biases_batch, dim=0).transpose(0, 1).flatten(0, 1)
        old_opes_appertain_batch = torch.stack(memory.opes_appertain_batch, dim=0).transpose(0, 1).flatten(0, 1)

        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                self.optimizer.zero_grad()
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx],
                                         old_ope_step_batch[start_idx: end_idx, :],
                                         old_end_ope_biases_batch[start_idx: end_idx, :],
                                         old_opes_appertain_batch[start_idx: end_idx, :])
                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                advantage = 0
                advantages = []
                for i_re in range(len(rewards_envs[i*minibatch_size:(i+1)*minibatch_size]) - 1):
                    delta = rewards_envs[i*minibatch_size + i_re] + self.gamma * state_values[i_re + 1].detach() - state_values[i_re].detach()
                    advantage = delta + self.gamma * self.gae_lambda * advantage
                    advantages.append(advantage)
                delta = rewards_envs[end_idx - 1] - state_values[end_idx - i*minibatch_size - 1].detach()
                advantage = delta + self.gamma * self.gae_lambda * advantage
                advantages.append(advantage)
                advantages = torch.tensor(advantages)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1, surr2) \
                       + self.vf_coeff * self.MseLoss(state_values,
                                                      rewards_envs[i * minibatch_size:(i + 1) * minibatch_size])
                loss_epochs += loss.mean().detach()

                loss.mean().backward()
                self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
