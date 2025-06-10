# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import jittor as jt

import numpy as np
from optim_scheduler import create_jittor_optimizer

import utils

def accuracy(output, target, topk=(1,)):
    """计算top-k准确率 - Jittor版本"""
    with jt.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]
        # 获取top-k预测
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.transpose(0, 1)
        
        # 扩展target维度并比较
        target_expanded = target.view(1, -1).expand_as(pred)
        correct = (pred == target_expanded)  # 使用Jittor的==操作符替代eq
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k * (100.0 / batch_size))
        
        return res

def index_fill(tensor, dim, index, value):
    """
    模拟PyTorch的index_fill功能
    
    Args:
        tensor: 输入张量
        dim: 要填充的维度
        index: 要填充的索引
        value: 填充值
    
    Returns:
        填充后的张量
    """
    # 创建索引元组
    indices = [slice(None)] * tensor.ndim
    indices[dim] = index
    
    # 复制张量以避免就地修改
    result = tensor.clone()
    result[tuple(indices)] = value
    
    return result


def train_one_epoch(model: jt.nn.Module, original_model: jt.nn.Module, 
                    criterion, data_loader: Iterable, optimizer, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None):
    
    
    model.train() if set_training_mode else model.eval()
    original_model.eval()
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    bactch_list = []
    for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # 数据预处理和检查
        input = input.float32()  # 确保数据类型正确
        target = target.int64()  # 确保标签类型正确
        # 检查输入数据
        if jt.isnan(input).any():
            print(f"警告: 输入数据在batch {batch_idx}中包含NaN")
            continue
            
        if jt.isinf(input).any():
            print(f"警告: 输入数据在batch {batch_idx}中包含Inf")
            continue 
        # 获取原始模型特征
        with jt.no_grad():
            if original_model is not None:
                try:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                    
                    # 检查特征是否正常
                    if jt.isnan(cls_features).any() or jt.isinf(cls_features).any():
                        print(f"警告: 原始模型特征异常在batch {batch_idx}")
                        cls_features = None
                except Exception as e:
                    print(f"原始模型推理错误: {e}")
                    cls_features = None
            else:
                cls_features = None
        
        # 前向传播
        try:
            output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
  
            logits = output['logits']
            # 检查logits是否正常
            if jt.isnan(logits).any():
                print(f"警告: logits包含NaN在batch {batch_idx}")
                print(f"模型参数检查:")
                check_model_params(model)
                continue
                
            if jt.isinf(logits).any():
                print(f"警告: logits包含Inf在batch {batch_idx}")
                continue
                
        except Exception as e:
            print(f"前向传播错误: {e}")
            continue
        # 应用类别掩码
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = jt.array(not_mask).int32()
            logits = index_fill(logits, dim=1, index=not_mask, value=-66.0)
        
        # 计算损失
        try:
            loss = criterion(logits, target)
            if loss<0:
                print(loss)
            # 添加额外的约束项
            if args.pull_constraint and 'reduce_sim' in output:
                constraint_loss = args.pull_constraint_coeff * output['reduce_sim']
                if not jt.isnan(constraint_loss) and not jt.isinf(constraint_loss):
                    loss = loss - constraint_loss + 2
                else:
                    print(f"警告: constraint_loss异常: {constraint_loss.item()}")
            
            # 检查损失是否正常
            if jt.isnan(loss) or jt.isinf(loss):
                print(f"损失异常在batch {batch_idx}: {loss.item()}")
                print(f"logits统计: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
                print(f"target统计: {target}")
                check_model_params(model)
                continue
                
        except Exception as e:
            print(f"损失计算错误: {e}")
            continue
        # 计算准确率
        try:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        except Exception as e:
            print(f"准确率计算错误: {e}")
            acc1, acc5 = jt.array(0.0), jt.array(0.0)
        bactch_list.append([loss.item(), acc1.item(), acc5.item()])
        if batch_idx % 50 == 0:  # 减少打印频率
            print(f"Batch {batch_idx}: loss: {loss.item():.6f}, acc1: {acc1.item():.2f}, acc5: {acc5.item():.2f}")
            # 保存这部分的数据
        # 检查损失是否为有限值
        if not math.isfinite(loss.item()):
            print(f"损失为 {loss.item()}，停止训练")
            print("最后的模型状态:")
            check_model_params(model)
            sys.exit(1)
        # 反向传播
        optimizer.zero_grad()
        
        try:
            optimizer.backward(loss)  
            optimizer.step()
        except Exception as e:
            print(f"反向传播错误: {e}")
            continue
        # 同步和更新指标
        jt.sync_all()
        metric_logger.update(Loss=loss.item())
        
        # metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("平均统计:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, bactch_list
def check_model_params(model):
    """检查模型参数是否异常"""
    print("=== 模型参数检查 ===")
    nan_count = 0
    inf_count = 0
    
    for name, param in model.named_parameters():
        if param is None:
            continue
            
        if jt.isnan(param).any():
            nan_count += 1
            print(f"参数包含NaN: {name}")
            
        if jt.isinf(param).any():
            inf_count += 1
            print(f"参数包含Inf: {name}")
            
        # 打印参数统计
        if nan_count + inf_count < 5:  # 避免输出太多
            print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    print(f"总计: {nan_count} 个NaN参数, {inf_count} 个Inf参数")
    print("=" * 50)



@jt.no_grad()
def evaluate(model: jt.nn.Module, original_model: jt.nn.Module, data_loader,
             task_id=-1, class_mask=None, args=None):
    criterion = jt.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with jt.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input
            target = target

            # compute output
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                mask = jt.array(mask).int32()
                logits_mask = jt.ones_like(logits) * float('-inf')
                logits_mask = index_fill(logits_mask, 1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@jt.no_grad()
def evaluate_till_now(model: jt.nn.Module, original_model: jt.nn.Module, data_loader, 
                    task_id=-1, class_mask=None, acc_matrix=None, args=None):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
        
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: jt.nn.Module, model_without_ddp: jt.nn.Module, original_model: jt.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: jt.nn.Optimizer, lr_scheduler, class_mask=None, args=None):
    
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    
    task_list = []
    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k
                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with jt.no_grad():
                        model.prompt.prompt.grad = None
                        model.prompt.prompt.assign(model.prompt.prompt.index_fill_(0, cur_idx, model.prompt.prompt.index_select(0, prev_idx)))
                        optimizer.param_groups[0]['params'] = model.parameters()
        
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k
                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with jt.no_grad():
                    model.prompt.prompt_key.grad = None
                    model.prompt.prompt_key.assign(model.prompt.prompt_key.index_fill_(0, cur_idx, model.prompt.prompt_key.index_select(0, prev_idx)))
                    optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_jittor_optimizer(args, model)
        
        epoch_list = []
        for epoch in range(args.epochs):  
            train_stats, batch_list = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)
            epoch_list.append(batch_list)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)
                
        task_list.append(epoch_list)
                
        #保存性能
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint/task{task_id+1}_checkpoint.pkl')
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and utils.is_main_process():
            log_file = os.path.join(args.output_dir, f"{datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M')}_stats.txt")
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    task_list_arr = np.array(task_list)
    np.save(os.path.join(args.output_dir, 'task_list_arr.npy'), task_list_arr)
     
