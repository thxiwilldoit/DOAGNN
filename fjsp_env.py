import copy
import json
import os
import sys
from dataclasses import dataclass
import random
import matplotlib.patches as mpatches
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import math


def read_json(path: str) -> dict:
    with open(path+".json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def write_json(data: dict, path: str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


def nums_detec(lines):
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes


def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0
    num_option = np.array([])
    mac = 0
    for i in line_split:
        x = int(i)
        if flag == 0:
            num_ope = x
            flag += 1
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope


def load_fjs(lines, num_mas, num_opes):
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []
    opes_appertain = np.array([])
    num_ope_biases = []
    for line in lines:
        if flag == 0:
            flag += 1
        elif line == "\n":
            break
        else:
            num_ope_bias = int(sum(nums_ope))
            num_ope_biases.append(num_ope_bias)
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    example: feat_job_batch[2, 2] opes_appertain_batch[0, 0, 1, 1] [2, 2].gather[1, [0, 0, 1, 1]]
    The index of index:[0, 0, 1, 1] is [0, 0][0, 1][0, 2][0, 3] (the index of index).
    Since the first digit is 1, the column index will be replaced.
    [0, 0][0, 1][0, 2][0, 3] => [0, 0][0, 0][0, 1][0, 1], which corresponds to the specific values in [2, 2], so the result is [2, 2, 2, 2].
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)


@dataclass
class EnvState:
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None
    done_ope_sequence: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time, done_ope_sequence):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time
        self.done_ope_sequence = done_ope_sequence


class FJSPEnv(gym.Env):
    def __init__(self, case, env_paras, data_source='case'):
        self.show_mode = env_paras["show_mode"]
        self.batch_size = env_paras["batch_size"]
        self.num_jobs = env_paras["num_jobs"]
        self.num_mas = env_paras["num_mas"]
        self.paras = env_paras
        self.device = env_paras["device"]
        self.reward_mood = env_paras["reward_mood"]
        self.reward_ratios = env_paras["reward_ratios"]
        self.free_time_memory = torch.zeros(self.batch_size)
        self.diff_max_min = 0
        num_data = 8
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        self.last_time = torch.zeros(self.batch_size)
        lines = []
        if data_source=='case':
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        else:
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)

                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)

        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()

        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        self.proc_time_mas = torch.zeros(self.batch_size, num_mas)

        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)

        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)

        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()

        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()

        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()

        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1

        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        self.done_ope_sequence = torch.full((self.batch_size, self.nums_opes[0]), -1)
        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size).int()

        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1), self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes, done_ope_sequence = self.done_ope_sequence)

        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]

        ratio = self.reward_ratios
        self.N += 1

        for i in range(opes.shape[0]):
            index = (self.done_ope_sequence[i] == -1).nonzero(as_tuple=True)[0]
            if index.numel() > 0:
                self.done_ope_sequence[i, index[0]] = opes[i]

        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.proc_time_mas[self.batch_idxes, mas] += proc_times
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() \
                         * un_scheduled
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                              self.batch_idxes, :])

        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]

        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],
                                                                          dim=1).float()

        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()
        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        if self.reward_mood == "r":
            self.reward_batch = self.makespan_batch - max
        elif self.reward_mood == "rb":
            diff_cur_time = self.time[self.batch_idxes, None] - self.last_time
            self.last_time = self.time[self.batch_idxes, None]
            self.reward_batch = diff_cur_time.squeeze()
        elif self.reward_mood == "rf":
            free_time = self.machines_batch[self.batch_idxes, :, 1] - self.machines_batch[self.batch_idxes, :, 2]
            all_free_time = torch.sum(free_time, dim=1)
            diff_free = all_free_time - self.free_time_memory
            self.free_time_memory = all_free_time.detach()
            self.reward_batch = -diff_free
        elif self.reward_mood == "rbf":
            cur_reward = (self.makespan_batch - max) * ratio[0]
            diff_cur_time = self.time - self.last_time
            self.last_time = self.time.detach()
            cur_reward += (diff_cur_time * ratio[1])
            free_time = self.machines_batch[:, :, 1] - self.machines_batch[:, :, 2]
            all_free_time = torch.sum(free_time, dim=1)
            diff_free = all_free_time - self.free_time_memory
            self.free_time_memory = all_free_time.detach()
            cur_reward -= (diff_free * ratio[2])
            self.reward_batch = cur_reward
        self.makespan_batch = max
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time, self.done_ope_sequence)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)

        a = self.machines_batch[:, :, 1]
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        c = torch.min(b, dim=1)[0]
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.proc_time_mas = torch.zeros(self.batch_size, self.num_mas)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.free_time_memory = torch.zeros(self.batch_size)
        self.last_time = torch.zeros(self.batch_size)
        self.done_ope_sequence = torch.full((self.batch_size, self.nums_opes[0]), -1)
        return self.state

    def render(self, mode='human', save_pth='./save_gantt', name_ins='try', test_sample=False, index_min=0):
        if self.show_mode == 'draw':
            if test_sample == False:
                os.makedirs(save_pth, exist_ok=True)
                num_jobs = self.num_jobs
                num_mas = self.num_mas
                color = read_json("./utils/color_config")["gantt_color"]
                plt.rcParams['font.family'] = 'Times New Roman'
                if len(color) < num_jobs:
                    num_append_color = num_jobs - len(color)
                    color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in range(num_append_color)]
                write_json({"gantt_color": color}, "./utils/color_config")
                for batch_id in range(self.batch_size):
                    schedules = self.schedules_batch[batch_id].to('cpu')
                    fig = plt.figure(figsize=(10, 6))
                    fig.canvas.manager.set_window_title('Visual_gantt')
                    axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                    y_ticks = []
                    y_ticks_loc = []
                    for i in range(num_mas):
                        y_ticks.append('$M_{0}$'.format(i + 1))
                        y_ticks_loc.insert(0, i)
                    labels = [''] * num_jobs
                    for j in range(num_jobs):
                        labels[j] = "$J_{{0}}$".format(j + 1)
                    patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                    axes.cla()
                    axes.grid(linestyle='-.', color='gray', alpha=0.2)
                    axes.set_xlabel('Time')
                    axes.set_ylabel('Machine')
                    axes.set_yticks(y_ticks_loc, y_ticks)
                    axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                    axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                    for i in range(int(self.nums_opes[batch_id])):
                        id_ope = i
                        idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                        id_machine = schedules[id_ope][1]
                        axes.barh((num_mas - 1 - int(id_machine)),
                                 schedules[id_ope][3] - schedules[id_ope][2],
                                 left=schedules[id_ope][2],
                                 color=color[idx_job],
                                 height=0.5)
                    path_all = save_pth + '/' + name_ins
                    plt.savefig(path_all)
                    plt.close()
            elif test_sample == True:
                os.makedirs(save_pth, exist_ok=True)
                num_jobs = self.num_jobs
                num_mas = self.num_mas
                color = read_json("./utils/color_config")["gantt_color"]
                plt.rcParams['font.family'] = 'Times New Roman'
                if len(color) < num_jobs:
                    num_append_color = int(num_jobs - len(color))
                    color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in range(num_append_color)]
                write_json({"gantt_color": color}, "./utils/color_config")
                schedules = self.schedules_batch[index_min].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.manager.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('$M_{0}$'.format(i + 1))
                    y_ticks_loc.insert(0, i)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "$J_{0}$".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in
                           range(self.num_jobs)]
                axes.cla()
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[index_min])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, index_min)
                    id_machine = schedules[id_ope][1]
                    axes.barh((num_mas - 1 - int(id_machine)),
                              schedules[id_ope][3] - schedules[id_ope][2],
                              left=schedules[id_ope][2],
                              color=color[idx_job],
                              height=0.5)
                path_all = save_pth + '/' + name_ins
                plt.savefig(path_all)
                plt.close()
        return

    def get_idx(self, id_ope, batch_id):
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
