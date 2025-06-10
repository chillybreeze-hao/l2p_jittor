import jittor as jt
import jittor.optim as optim
import math
def create_jittor_optimizer(args, model):
    """
    根据args参数创建适配的优化器
    特别针对L2P (Learning to Prompt) 模型
    """
    # 获取参数
    opt_name = args.opt.lower()  # 'adam'
    lr = args.lr  # 0.03
    weight_decay = args.weight_decay  # 0.0
    
    # 处理冻结参数
    freeze_layers = getattr(args, 'freeze', [])
    trainable_params = []
    frozen_params = []
    
    # print(f"冻结的层: {freeze_layers}")
    
    if freeze_layers:
        for name, param in model.named_parameters():
            # 检查参数是否在冻结列表中
            should_freeze = False
            for freeze_name in freeze_layers:
                if freeze_name in name:
                    should_freeze = True
                    break
            
            if should_freeze:
                param.stop_grad()  # Jittor中停止梯度计算
                frozen_params.append(name)
                # print(f"冻结参数: {name}")
            else:
                trainable_params.append(param)
                # print(f"可训练参数: {name}")
        
        # print(f"总共冻结 {len(frozen_params)} 个参数")
        # print(f"总共可训练 {len(trainable_params)} 个参数")
        
        # 只对可训练参数创建优化器
        parameters = trainable_params
    else:
        parameters = model.parameters()
    
    # print(f"创建优化器: {opt_name}, 学习率: {lr}, 权重衰减: {weight_decay}")
    
    if opt_name == 'adam':
        betas = getattr(args, 'opt_betas', (0.9, 0.999))  # (0.9, 0.999)
        eps = getattr(args, 'opt_eps', 1e-8)  # 1e-08
        optimizer = optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
    elif opt_name == 'adamw':
        betas = getattr(args, 'opt_betas', (0.9, 0.999))
        eps = getattr(args, 'opt_eps', 1e-8)
        optimizer = optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
    elif opt_name == 'sgd':
        momentum = getattr(args, 'momentum', 0.9)  # 0.9
        nesterov = getattr(args, 'nesterov', False)
        optimizer = optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
    else:
        raise ValueError(f"不支持的优化器: {opt_name}")
    
    return optimizer

# 需要先定义JittorScheduler基类
class JittorScheduler:
    """Jittor学习率调度器基类"""
    
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def get_lr(self):
        """计算当前学习率，子类需要实现"""
        raise NotImplementedError
        
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
# 需要定义其他依赖的调度器类
class StepLR(JittorScheduler):
    """步长学习率调度器"""
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * (self.gamma ** (self.last_epoch // self.step_size))
                for base_lr in self.base_lrs]
class LinearLR(JittorScheduler):
    """线性学习率调度器"""
    
    def __init__(self, optimizer, start_factor=1.0, end_factor=0.0, total_iters=5, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        
        if self.last_epoch > self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
            
        factor = (self.end_factor * self.last_epoch + 
                 self.start_factor * (self.total_iters - self.last_epoch)) / self.total_iters
        return [base_lr * factor for base_lr in self.base_lrs]
class WarmupScheduler(JittorScheduler):
    """带热身的学习率调度器"""
    
    def __init__(self, optimizer, warmup_scheduler, main_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        super(WarmupScheduler, self).__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_lr()
        else:
            # 调整主调度器的epoch
            self.main_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            return self.main_scheduler.get_lr()
            
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.last_epoch = epoch
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.last_epoch = epoch - self.warmup_epochs
            self.main_scheduler.step()
            
        # 应用学习率
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
# 现在是你的调度器类
class ConstantLR(JittorScheduler):
    """常数学习率调度器"""
    
    def __init__(self, optimizer, factor=1.0, last_epoch=-1):
        self.factor = factor
        super(ConstantLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [base_lr * self.factor for base_lr in self.base_lrs]
class ConstantWithWarmupLR(JittorScheduler):
    """带热身的常数学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super(ConstantWithWarmupLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性热身
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                   for base_lr in self.base_lrs]
        else:
            # 常数学习率
            return self.base_lrs
class CosineAnnealingWithWarmupLR(JittorScheduler):
    """带热身的余弦退火调度器"""
    
    def __init__(self, optimizer, T_max, warmup_epochs=0, warmup_start_lr=1e-6, 
                 eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性热身
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                   for base_lr in self.base_lrs]
        else:
            # 余弦退火
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_T_max = self.T_max - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * cos_epoch / cos_T_max)) / 2
                    for base_lr in self.base_lrs]
def create_jittor_scheduler(args, optimizer):
    """
    根据args参数创建适配的学习率调度器
    """
    sched_name = args.sched.lower()  # 'constant'
    epochs = args.epochs  # 5
    warmup_epochs = getattr(args, 'warmup_epochs', 5)  # 5
    warmup_lr = getattr(args, 'warmup_lr', 1e-6)  # 1e-06
    min_lr = getattr(args, 'min_lr', 1e-5)  # 1e-05
    
    # print(f"创建学习率调度器: {sched_name}")
    # print(f"总轮次: {epochs}, 热身轮次: {warmup_epochs}")
    # print(f"热身学习率: {warmup_lr}, 最小学习率: {min_lr}")
    
    if sched_name == 'constant':
        if warmup_epochs > 0:
            scheduler = ConstantWithWarmupLR(
                optimizer, 
                warmup_epochs=warmup_epochs, 
                warmup_start_lr=warmup_lr
            )
        else:
            scheduler = ConstantLR(optimizer)
            
    elif sched_name == 'cosine':
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            T_max=epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_lr,
            eta_min=min_lr
        )
        
    elif sched_name == 'step':
        decay_epochs = getattr(args, 'decay_epochs', 30)  # 30
        decay_rate = getattr(args, 'decay_rate', 0.1)  # 0.1
        
        if warmup_epochs > 0:
            # 创建带热身的步长调度器
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_lr/optimizer.param_groups[0]['lr'],
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            main_scheduler = StepLR(optimizer, step_size=decay_epochs, gamma=decay_rate)
            scheduler = WarmupScheduler(optimizer, warmup_scheduler, main_scheduler, warmup_epochs)
        else:
            scheduler = StepLR(optimizer, step_size=decay_epochs, gamma=decay_rate)
            
    else:
        print(f"警告: 调度器 '{sched_name}' 未实现，使用常数调度器")
        if warmup_epochs > 0:
            scheduler = ConstantWithWarmupLR(
                optimizer, 
                warmup_epochs=warmup_epochs, 
                warmup_start_lr=warmup_lr
            )
        else:
            scheduler = ConstantLR(optimizer)
    
    return scheduler
#   创建学习率调度器

