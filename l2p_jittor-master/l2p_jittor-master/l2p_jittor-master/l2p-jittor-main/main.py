# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import jittor as jt
from pathlib import Path
from timm.models import create_model
from optim_scheduler import create_jittor_scheduler, create_jittor_optimizer
from datasets import build_continual_dataloader
from engine import *
from models import vit_tiny_patch16_224
import utils
from vision_transformer_jittor import VisionTransformer_jittor
from convert_pytorch_to_jittor import convert_pytorch_to_jittor_with_analysis
import warnings

warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
import os

def main(args):
    utils.init_distributed_mode(args)
    # fix the seed for reproducibility
    seed = args.seed
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    data_loader, class_mask = build_continual_dataloader(args)

    # args.nb_classes
    print(f"Creating original model: {args.model}")
    base_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )
    print(f"Number of parameters in original PyTorch model: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")
    model = VisionTransformer_jittor(num_classes=args.nb_classes,
                                     drop_rate=args.drop,
                                     drop_path_rate=args.drop_path)

    # 转换为基础Jittor模型（共享权重）
    print("Converting PyTorch model to Jittor model...")
    original_model = convert_pytorch_to_jittor_with_analysis(base_model, model)
    # 输出转换后的模型参数量：  
    n_parameters_original_model = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    print(f"Number of parameters in original Jittor model: {n_parameters_original_model}")
    del base_model  # 立即释放PyTorch模型
    del model  # 立即释放Jittor模型

    base_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    model_1 = VisionTransformer_jittor(num_classes=args.nb_classes,          
                                    drop_rate=args.drop,                   
                                    drop_path_rate=args.drop_path,         
                                    prompt_length=args.length,             
                                    embedding_key=args.embedding_key,     
                                    prompt_init=args.prompt_key_init,
                                    prompt_pool=args.prompt_pool,         
                                    prompt_key=args.prompt_key,           
                                    pool_size=args.size,                  
                                    top_k=args.top_k,                    
                                    batchwise_prompt=args.batchwise_prompt, 
                                    prompt_key_init=args.prompt_key_init,  
                                    head_type=args.head_type,              
                                    use_prompt_mask=args.use_prompt_mask)
    
    model = convert_pytorch_to_jittor_with_analysis(base_model, model_1)
    del base_model  # 立即释放PyTorch模型
    del model_1  # 立即释放Jittor模型
    # 输出转换后的模型参数量：
    n_parameters_converted_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in converted Jittor model: {n_parameters_converted_model}")
    
    jt.flags.use_cuda = 1
    
    if args.freeze:
        # Jittor版 - 冻结original_model的所有参数
        original_model.requires_grad_(False)
        
        # Jittor版 - 冻结指定前缀的参数
        for n, p in model.named_parameters():
            if any(n.startswith(prefix) for prefix in args.freeze):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                # 加载模型参数
                checkpoint = jt.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    
    if args.distributed:
        print("Can't use DistributedDataParallel")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_jittor_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler = create_jittor_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    # 交叉熵函数
    criterion = jt.nn.CrossEntropyLoss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_l2p':
        from configs.cifar100_l2p import get_args_parser
        config_parser = subparser.add_parser('cifar100_l2p', help='Split-CIFAR100 L2P configs')
    elif config == 'five_datasets_l2p':
        from configs.five_datasets_l2p import get_args_parser
        config_parser = subparser.add_parser('five_datasets_l2p', help='5-Datasets L2P configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)