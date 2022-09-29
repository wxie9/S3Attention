"""
"" Adapted from https://github.com/mlpen/Nystromformer
"" 
"""

from fvcore.nn import FlopCountAnalysis
from model_wrapper import ModelForSC, ModelForSCDual, ModelForSCProbing, ModelForSCDualProbing
from dataset import LRADataset
import torch
import random

from torch.utils.data import DataLoader

from torch.utils import data as torch_data


import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np

import math
import itertools
from config.listops.base_config import parser

from torch.utils.tensorboard import SummaryWriter   
from time import gmtime, strftime, localtime


# import tqdm
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

def step(component, step_idx, accumu_steps, amp_scaler=None,dev_step_idx_cumu=None):
    t0 = time.time()
    
    _, batch = next(ds_iter[component])
    # if component == 'train':
        # print('我也不知道我在干啥', _, step_idx,'\n')
    for key in batch:
        batch[key] = batch[key].cuda()
    if (args.model == 'nystrom' or args.model == 'reformer') and args.pooling_mode.lower() == 'cls':
        for key in batch:
            if 'input_ids' in key or 'mask' in key:
                batch[key] = batch[key][:, :-1].contiguous()
    outputs = {}
    partial_inputs_list = [{} for _ in range(accumu_steps)]
    for key in batch:
        for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
            partial_inputs_list[idx][key] = inp
   

    if component == "train":
        optimizer.zero_grad()
        # print('asdfasdf')
        # for partial_inputs in partial_inputs_list:
            # partial_outputs = model(**partial_inputs)
                    
                    # print(partial_inputs)
        for partial_inputs in partial_inputs_list:
            # print(**partial_inputs)
            # print(*partial_inputs, partial_inputs)
            if args.test_flops:
                # print('asdfasdf')
                # partial_inputs = {}
                # for p in partial_inputs_list:
            # partial_outputs = model(**partial_inputs)
                    # partial_inputs.update(p) 
                    # print(partial_inputs)
                if 'input_ids_1' in partial_inputs:
                    flops = FlopCountAnalysis(
                        model, [partial_inputs['input_ids_0'][:1], partial_inputs['input_ids_1'][:1],
                                partial_inputs['mask_0'][:1], partial_inputs['mask_1'][:1], partial_inputs['label'][:1]])
                else:
                    flops = FlopCountAnalysis(
                        model, [partial_inputs['input_ids_0'][:1], partial_inputs['mask_0'][:1], partial_inputs['label'][:1]])

                print(f"Flops of {args.model}: {flops.total()/1e9:.2f} G")
                exit()  
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

            
            
                
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            
                
        amp_scaler.step(optimizer)
        amp_scaler.update()
        if (not args.fixed_lr) or step_idx < args.warmup:
        #     lr_scheduler.step()
        # if  step_idx < args.warmup:
            lr_scheduler.step()
            
    else:
        with torch.no_grad():
            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()
    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)
    
    
    
    if component == "train":
        writer.add_scalar('train_loss', loss, step_idx, walltime = time_since_start)
        writer.add_scalar('train_accu', accu, step_idx, walltime = time_since_start)
        writer.add_scalar('learning_rate', learning_rate, step_idx, walltime = time_since_start)
    elif component == "dev" or component == "dev_test":
        writer.add_scalar('test_loss', loss, step_idx+dev_step_idx_cumu, walltime = time_since_start)
        writer.add_scalar('test_accu', accu, step_idx+dev_step_idx_cumu, walltime = time_since_start)
    
    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)
    
    
    
    

def print_summary(summary, save_if_improved, train_step_idx, subset):
    # subset: str, the subset to report the result
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key+f"_{subset}"] = summary[key]
        else:
            summary_round[key+f"_{subset}"] = round(summary[key], 4)

    print(summary_round, flush=True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []
    
    
    


    
    
if __name__ == "__main__":
    args = parser.parse_args()

    # Freeze random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Load model and task, need updated later
    args.attn_type = args.model # remove attn_type in the future
    args.mixed_precision = False # bool(args.mixed_precision)
    task = args.task



    cur_time = strftime("%Y-%m-%d_%H:%M:%S", localtime())
    pwd = os.getcwd()
    log_path = f'{pwd}/logs/log_{args.task}_{args.model}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(f'{log_path}/exp_lr_{args.learning_rate}_{cur_time}')



    checkpoint_dir = args.chk_path
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if args.local_rank ==0:
        print(args)

    device_ids = list(range(torch.cuda.device_count()))
    if args.local_rank ==0:
        print(f"GPU list: {device_ids}")




    if task == "retrieval":
        if args.test_flops:
            model = ModelForSCDualProbing(args)
        else:
            model = ModelForSCDual(args)
    else:
        if args.test_flops:
            model = ModelForSCProbing(args)
        else:
            model = ModelForSC(args)

    if args.local_rank == 0:
        print(model)
        print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
        print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush=True)
    # nn.init.normal_(model.parameters)sparse_

    
    
    
    torch.distributed.init_process_group(backend="nccl")
    local_device = torch.device("cuda:{}".format(args.local_rank))
    
    
    # [nn.init.normal_(weight,std = 1e-2) for weight in model.parameters()]
    # [nn.init.uniform_(weight,a = -1e-1,b = 1e-1) for weight in model.parameters()]
    
    model = model.to(local_device)#.cuda()
    # model = nn.DataParallel(model, device_ids = device_ids)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids = [local_device],find_unused_parameters=True)

    # data_path = '/home/xue.w/cwq/transformer-ls/lra/datasets'
    data_path = '../datasets'
    ds_iter = {
        "train":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.train.pickle", True), batch_size=args.batch_size, drop_last=True)),
        "dev":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.dev.pickle", True), batch_size=args.batch_size, drop_last=True)),
        
        "dev_test":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", True), batch_size=args.batch_size, drop_last=True)),
        "test":enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size*2, drop_last=True)),
    }

    train_dataset = LRADataset(f"{data_path}/{task}.train.pickle", False)  
    sampler = torch_data.distributed.DistributedSampler(train_dataset)
    Train_Data = torch_data.DataLoader(train_dataset,batch_size=args.batch_size,
                                sampler=sampler, num_workers=4,drop_last=True) 
    
    
    # Train_Data =  DataLoader(, batch_size=args.batch_size, drop_last=True)
    Validate_Data =  DataLoader(LRADataset(f"{data_path}/{task}.dev.pickle", False), batch_size=args.batch_size*4, drop_last=False)
    Test_Data = DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size*4, drop_last=False)
    
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999), eps=args.adam_eps, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.learning_rate,
        pct_start=args.warmup / args.num_train_steps,
        anneal_strategy=args.lr_decay,
        total_steps=args.num_train_steps
    )

  
    
    
    
    amp_scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    
    init_t = time.time()

    log_f_path = os.path.join(checkpoint_dir, f"{args.expname}_output.log")
    log_f = open(log_f_path, "a+")

    summary = {
        component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test","dev_test"]
    }

    # accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
    accumu_steps = max(args.batch_size // len(device_ids) // args.global_step, 1)
    print(f"accumu_steps={accumu_steps}")

    
    
#     dev_step_idx_cumu = 0
    
#     if args.skip_train == 0:
#         try:
#             model.train()
#             for train_step_idx in range(args.num_train_steps):
#                 outputs = step("train", train_step_idx,accumu_steps=accumu_steps,amp_scaler=amp_scaler)

#                 if (train_step_idx + 1) % args.eval_frequency == 0:
#                     print_summary(summary["train"], False, train_step_idx, 'train')
#                     model.eval()
#                     for dev_step_idx in range(args.num_eval_steps):
#                         outputs = step("dev", dev_step_idx,
#                                        accumu_steps=accumu_steps,
#                                        amp_scaler=amp_scaler,
#                                        dev_step_idx_cumu=dev_step_idx_cumu
#                                       )
#                     print_summary(summary["dev"], True, train_step_idx, 'dev')
# #                     for dev_step_idx in range(args.num_eval_steps):
# #                         outputs = step("dev_test", dev_step_idx,
# #                                        accumu_steps=accumu_steps,
# #                                        amp_scaler=amp_scaler,
# #                                        dev_step_idx_cumu=dev_step_idx_cumu
# #                                       )
# #                     print_summary(summary["dev_test"], True, train_step_idx, 'dev_test')
#                     dev_step_idx_cumu = dev_step_idx_cumu + args.num_eval_steps
#                     #for test_step_idx in range(39):
#                     #    outputs = step("test", test_step_idx)
#                     #print_summary(summary["test"], False, train_step_idx, 'test')
#                     model.train()
#         except KeyboardInterrupt as e:
#             print(e)


    loss_cumu_global = 0
    accu_cumu_global = 0      
    best_accu = 0
    best_loss = 1e7
            
    for epoch in range(args.epoch):
        model.train()
        # loss_cumu_global = 0
        # accu_cumu_global = 0    
        loss_cumu = 0
        accu_cumu = 0
        if args.local_rank == 0:
            with tqdm(Train_Data, unit="batch") as tepoch:
                for step_idx, batch in enumerate(tepoch, 0):#enumerate(iterator):    
                    tepoch.set_description(f"Epoch {epoch}")

                    for key in batch:
                        batch[key] = batch[key].cuda()
                    outputs = {}
                    optimizer.zero_grad()
                    partial_outputs = model(**batch)
                    for key in partial_outputs:
                        partial_outputs[key] = partial_outputs[key].mean()
                        if key not in outputs:
                            outputs[key] = partial_outputs[key]
                        else:
                            outputs[key] += partial_outputs[key]
                    if amp_scaler is not None:
                        amp_scaler.scale(partial_outputs["loss"]).backward()
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                    else:
                        optimizer.zero_grad()
                        partial_outputs["loss"].backward()
                        optimizer.step()
                    if (not args.fixed_lr) or step_idx+epoch*len(Train_Data) < args.warmup:
                        lr_scheduler.step()


                    if np.mod(step_idx+epoch*len(Train_Data),args.eval_frequency) == 0 and args.local_rank == 0:
                        loss_val = 0
                        accu_val = 0
                        with torch.no_grad():
                            for batch_val in Validate_Data:
                                for key in batch_val:
                                    batch_val[key] = batch_val[key].cuda()
                                outputs_val = {}
                                partial_outputs = model(**batch_val)

                                for key in partial_outputs:
                                    partial_outputs[key] = partial_outputs[key].mean() 
                                    if key not in outputs_val:
                                        outputs_val[key] = partial_outputs[key]
                                    else:
                                        outputs_val[key] += partial_outputs[key]
                                loss_val = loss_val + outputs_val['loss'].data.item()
                                accu_val = accu_val + outputs_val['accu'].data.item()
                    batch_size = batch[list(batch.keys())[0]].size(0)

                    learning_rate = optimizer.param_groups[0]["lr"]
                    loss = outputs["loss"].data.item()
                    accu = outputs["accu"].data.item()
                    time_since_start = time.time() - init_t


                    loss_cumu = loss_cumu + loss
                    loss_cumu_global = loss_cumu_global + loss
                    accu_cumu = accu_cumu + accu
                    accu_cumu_global = accu_cumu_global +accu
                    
                    writer.add_scalar('train_loss', loss, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                    writer.add_scalar('train_accu', accu, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                    writer.add_scalar('learning_rate', learning_rate, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                    
                    # writer.add_scalar('train_loss', loss_cumu_global /(step_idx+epoch*len(Train_Data)+1), step_idx+epoch*len(Train_Data), walltime = time_since_start)
                    # writer.add_scalar('train_accu', accu_cumu_global / (step_idx+epoch*len(Train_Data)+1), step_idx+epoch*len(Train_Data), walltime = time_since_start)
                    # writer.add_scalar('learning_rate', learning_rate, step_idx+epoch*len(Train_Data), walltime = time_since_start)

                    tepoch.set_postfix(loss=loss_cumu /(step_idx+1), accu=accu_cumu /(step_idx+1)
                                      ,val_loss= loss_val/len(Validate_Data), val_accu = accu_val/len(Validate_Data)
                                      )

                    # tepoch.set_postfix(loss=loss_cumu /(step_idx+1), accu=accu_cumu /(step_idx+1))


                    if best_accu < accu_val / len(Validate_Data):
                        best_accu = accu_val / len(Validate_Data)
                    # # best_accu = summary["best_accu"]
                        torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
                        print(f"best_accu={best_accu}. Saved best model")
                        
                        
#                     if best_loss > loss_val/len(Validate_Data):
#                         best_loss  =  loss_val/len(Validate_Data)
#                     # # best_accu = summary["best_accu"]
#                         torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
#                         print(f"best_loss={best_loss}. Saved best model")
                        
                        
                    


        else:
            for step_idx, batch in enumerate(Train_Data, 0):

                for key in batch:
                    batch[key] = batch[key].cuda()
                outputs = {}
                optimizer.zero_grad()
                partial_outputs = model(**batch)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean()
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]
                if amp_scaler is not None:
                    amp_scaler.scale(partial_outputs["loss"]).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.zero_grad()
                    partial_outputs["loss"].backward()
                    optimizer.step()
                if (not args.fixed_lr) or step_idx+epoch*len(Train_Data) < args.warmup:
                    lr_scheduler.step()
                    
                    
                batch_size = batch[list(batch.keys())[0]].size(0)

                learning_rate = optimizer.param_groups[0]["lr"]
                loss = outputs["loss"].data.item()
                accu = outputs["accu"].data.item()
                time_since_start = time.time() - init_t


                loss_cumu = loss_cumu + loss
                loss_cumu_global = loss_cumu_global + loss
                accu_cumu = accu_cumu + accu
                accu_cumu_global = accu_cumu_global +accu
                writer.add_scalar('train_loss', loss, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                writer.add_scalar('train_accu', accu, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                writer.add_scalar('learning_rate', learning_rate, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                # writer.add_scalar('train_loss', loss_cumu_global /(step_idx+epoch*len(Train_Data)+1), step_idx+epoch*len(Train_Data), walltime = time_since_start)
                # writer.add_scalar('train_accu', accu_cumu_global / (step_idx+epoch*len(Train_Data)+1), step_idx+epoch*len(Train_Data), walltime = time_since_start)
                # writer.add_scalar('learning_rate', learning_rate, step_idx+epoch*len(Train_Data), walltime = time_since_start)
                
                
     

        if args.local_rank == 0:# and epoch % 10 == 0:
            
            torch.save({"model_state_dict_Train":model.module.state_dict()}, log_f_path.replace(".log", ".model.train"))

            ds_iter["test"] = enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size*8, drop_last=True))


            checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
            # print(checkpoint)
            model.module.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            try:
                for test_step_idx in itertools.count():
                    outputs = step("test", test_step_idx,accumu_steps=accumu_steps)
            except StopIteration:
                print_summary(summary["test"], False, step_idx+epoch*len(Train_Data), 'test')
                
                
                

            
            ds_iter["test"] = enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size*8, drop_last=True))
            checkpoint = torch.load(log_f_path.replace(".log", ".model.train"), map_location="cpu")
            model.module.load_state_dict(checkpoint["model_state_dict_Train"])
            model.eval()
            try:
                for test_step_idx in itertools.count():
                    outputs = step("test", test_step_idx,accumu_steps=accumu_steps)
            except StopIteration:
                print_summary(summary["test"], False, step_idx+epoch*len(Train_Data), 'test')
                
            # checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
           

        
        
        
        
        
        
      
#         optimizer.zero_grad()
#         output = model(src, trg)
#         output_dim = output.shape[-1]

#         output = output[1:].view(-1, output_dim)
#         trg = trg[1:].view(-1)
        
#         loss = criterion(output, trg)

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

#         optimizer.step()

#         epoch_loss += loss.item()
        

            
    
    
    ds_iter["test"] = enumerate(DataLoader(LRADataset(f"{data_path}/{task}.test.pickle", False), batch_size=args.batch_size*4, drop_last=True))

    checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
    model.module.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    try:
        for test_step_idx in itertools.count():
            outputs = step("test", test_step_idx,accumu_steps=accumu_steps)
    except StopIteration:
        print_summary(summary["test"], False, step_idx+epoch*len(Train_Data), 'test')
    
    
    