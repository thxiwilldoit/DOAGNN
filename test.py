import copy
import json
import os
import random
import time as time
import gym
import pandas as pd
import pynvml
import PPO_model
import torch
import numpy as np


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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    print(torch.__version__)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ope"]
    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir('./model/')[:]
    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_types=env_paras["num_mas"])
    rules = test_paras["rules"]
    envs = []
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/test_{0}'.format(str_time)
    gantt_save_pth = save_path + '/save_gantt'
    os.makedirs(save_path)
    writer = pd.ExcelWriter(
        '{0}/makespan_{1}.xlsx'.format(save_path, str_time))
    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))
    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])
    data_file.to_excel(writer, sheet_name='Sheet1', index=False)
    writer._save()
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
    writer_time._save()
    start = time.time()

    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        gantt_save_pth_n = gantt_save_pth + '/' + os.path.splitext(rule)[0]
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location=device)
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            for name, param in model_CKPT.items():
                if name == 'get_fir_operations.0.coeff':
                    para = model.policy.state_dict()["get_fir_operations.0.coeff"]
                    para_shape = para.shape
                    pt_shape = model_CKPT[name].shape
                    if para_shape[0] > pt_shape[0]:
                        new_param = torch.zeros_like(para)
                        new_param[:para_shape[0], :] = para
                        model_CKPT[name] = new_param
                    elif para_shape[0] < pt_shape[0]:
                        model_CKPT[name] = model_CKPT[name][:para_shape[0], :]
                elif name == 'get_fir_operations.0.fc_ope_ma.0.weight':
                    para = model.policy.state_dict()["get_fir_operations.0.fc_ope_ma.0.weight"]
                    para_shape = para.shape
                    pt_shape = model_CKPT[name].shape
                    if para_shape[1] > pt_shape[1]:
                        new_param = torch.zeros_like(para)
                        new_param[:, :para_shape[1]] = para
                        model_CKPT[name] = new_param
                    elif para_shape[1] < pt_shape[1]:
                        model_CKPT[name] = model_CKPT[name][:, :para_shape[1]]
            model.policy.load_state_dict(model_CKPT, strict=False)
            model.policy_old.load_state_dict(model_CKPT, strict=False)
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas
            if len(envs) == num_ins:
                env = envs[i_ins]
            else:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                else:
                    env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(os.path.splitext(test_files[i_ins])[0]))

            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
                finish_time = time.time()
                print("makespan is {}".format(makespans))
                print(makespan)
                makespan_list = makespan.clone().to(torch.device('cpu')).numpy().tolist()
                min_index = makespan_list.index(min(makespan_list))
                env.render(mode="human", save_pth=gantt_save_pth_n,
                                          name_ins=os.path.splitext(test_files[i_ins])[0], test_sample=True, index_min=min_index)
                txt_name = os.path.splitext(test_files[i_ins])[0] + '.txt'
                result_schedules = env.schedules_batch[min_index].clone().to(torch.device('cpu')).numpy()
                result_schedules = result_schedules.reshape((-1, result_schedules.shape[-1]))
                np.savetxt(os.path.join(gantt_save_pth_n, txt_name), result_schedules, fmt="%.2f")
                with open(os.path.join(gantt_save_pth_n, txt_name), "a") as file:
                    file.write('=' * 10 + '\n')
                    for item in line:
                        file.write(item + "\n")
            else:
                time_s = []
                makespan_s = []
                envs_best_makespan = copy.deepcopy(env)
                for j in range(test_paras["num_average"]):
                    makespan, time_re = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    if makespan == min(makespan_s):
                        envs_best_makespan = copy.deepcopy(env)
                    env.reset()
                finish_time = time.time()
                print("makespan is {}".format(float(min(makespan_s))))
                print(makespan_s)
                envs_best_makespan.render(mode="human", save_pth=gantt_save_pth_n, name_ins=os.path.splitext(test_files[i_ins])[0])
                txt_name = os.path.splitext(test_files[i_ins])[0] + '.txt'
                result_schedules = envs_best_makespan.schedules_batch.clone().to(torch.device('cpu')).numpy()
                result_schedules = result_schedules.reshape((-1, result_schedules.shape[-1]))
                np.savetxt(os.path.join(gantt_save_pth_n, txt_name), result_schedules, fmt="%.2f")
                with open(os.path.join(gantt_save_pth_n, txt_name), "a") as file:
                    file.write('=' * 10 + '\n')
                    for item in line:
                        file.write(item + "\n")
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
            print("finish env [{0}]".format(os.path.splitext(test_files[i_ins])[0]))
        print("rule_spend_time: ", finish_time - step_time_last)
        data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer._save()
        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer_time._save()
        for env in envs:
            env.reset()
    writer._save()
    writer.close()
    writer_time._save()
    writer_time.close()
    print("total_spend_time: ", time.time() - start)

def schedule(env, model, memories, flag_sample=False):
    state = env.state
    dones = env.done_batch
    done = False
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    spend_time = time.time() - last_time
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()
