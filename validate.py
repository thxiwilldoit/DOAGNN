import gym
import PPO_model
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    file_path = "./data_dev/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
