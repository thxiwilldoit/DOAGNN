import copy
import json
import os
import random
import time
from collections import deque
import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom
import PPO_model
from case_generator import CaseGenerator
from validate import validate, get_validate_env

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
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
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ope"]
    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"], num_types=num_mas)
    env_valid = get_validate_env(env_valid_paras)
    maxlen = 1
    best_models = deque()
    makespan_best = float('inf')
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    gantt_save_pth_n = save_path + '/save_gantt'
    os.makedirs(save_path)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    writer_100 = pd.ExcelWriter('{0}/training_{1}_{2}.xlsx'.format(save_path, env_valid_paras["batch_size"], str_time))
    valid_results = []
    valid_results_100 = []
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    writer_ave._save()
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    writer_100._save()
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):
        print("=================================" + str(i) + "=================================")
        if (i - 1) % train_paras["parallel_iter"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            print('The scale of the training instances are {} job(s) and {} machine(s).'.format(num_jobs, num_mas))
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
        env.render(mode="human", save_pth=gantt_save_pth_n, name_ins=str(i))
        print("Spend time: ", time.time()-last_time)
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        env.reset()
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("Get reward:", '%.3f' % reward, "and loss:", '%.3f' % loss)
            # wandb.log({"loss": loss, "reward": reward})
            # wandb.log({"grad_norm": (self.policy.Histogram.named_parameters())})
            # for name, param in model.policy.named_parameters():
            #     # print(name)
            #     # print(param.grad)
            #     if param.grad is not None:
            #         wandb.log({f'gradients/{name}': param.grad})

            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated
        if i % train_paras["save_timestep"] == 0:
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)
            if vali_result < makespan_best:
                makespan_best = vali_result
                print("-----THE {}th ITERATION IS THE BEST-----".format(i))
                print("The sum makespan is {}".format(vali_result*15))
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/model_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)
            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave._save()
    writer_ave.close()
    column = [i_col for i_col in range(env_valid_paras["batch_size"])]
    data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
    writer_100._save()
    writer_100.close()
    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()
