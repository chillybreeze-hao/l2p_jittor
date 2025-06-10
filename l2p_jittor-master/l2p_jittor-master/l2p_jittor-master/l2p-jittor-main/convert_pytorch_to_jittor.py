def print_jittor_model_weights(jittor_model):
    """打印Jittor模型的所有权重信息"""
    
    print("="*80)
    print("JITTOR模型权重详情")
    print("="*80)
    
    total_params = 0
    
    for name, param in jittor_model.named_parameters():
        param_count = param.numel() if hasattr(param, 'numel') else param.size
        total_params += param_count
        
        print(f"层名称: {name}")
        print(f"  形状: {param.shape}")
        print(f"  类型: {param.dtype}")
        print(f"  参数数量: {param_count:,}")
        print(f"  数值范围: [{param.min().item():.6f}, {param.max().item():.6f}]")
        print(f"  数值均值: {param.mean().item():.6f}")
        print(f"  数值标准差: {param.std().item():.6f}")
        print("-" * 50)
    
    print(f"总参数数量: {total_params:,}")
    print("="*80)
def print_model_structure_and_weights(jittor_model):
    """分层次显示模型结构和权重"""
    
    print("="*80)
    print("JITTOR模型结构与权重详情")
    print("="*80)
    
    def print_module_weights(module, module_name="", indent=0):
        """递归打印模块权重"""
        prefix = "  " * indent
        
        # 打印当前模块信息
        if hasattr(module, '_parameters') and module._parameters:
            print(f"{prefix}📁 {module_name} ({type(module).__name__})")
            
            for param_name, param in module._parameters.items():
                if param is not None:
                    print(f"{prefix}  ⚙️  {param_name}: {param.shape} ({param.dtype})")
                    print(f"{prefix}     参数量: {param.numel():,}")
                    print(f"{prefix}     范围: [{param.min().item():.4f}, {param.max().item():.4f}]")
                    print(f"{prefix}     均值: {param.mean().item():.4f}")
        
        # 递归处理子模块
        if hasattr(module, '_modules') and module._modules:
            for child_name, child_module in module._modules.items():
                if child_module is not None:
                    full_name = f"{module_name}.{child_name}" if module_name else child_name
                    print_module_weights(child_module, full_name, indent + 1)
    
    print_module_weights(jittor_model, "model")
    print("="*80)
def analyze_specific_layers(jittor_model):
    """分析特定层的权重"""
    
    print("="*80)
    print("关键层权重分析")
    print("="*80)
    
    # 分析PatchEmbed层
    if hasattr(jittor_model, 'patch_embed'):
        print("🔍 PATCH EMBED 层分析:")
        patch_embed = jittor_model.patch_embed
        
        if hasattr(patch_embed, 'proj'):
            proj = patch_embed.proj
            print(f"  Conv2d权重: {proj.weight.shape}")
            print(f"  Conv2d配置: in_channels={proj.in_channels}, out_channels={proj.out_channels}")
            print(f"  kernel_size: {proj.kernel_size}, stride: {proj.stride}")
            print(f"  权重统计: 最小值={proj.weight.min().item():.6f}, 最大值={proj.weight.max().item():.6f}")
            
            if proj.bias is not None:
                print(f"  偏置: {proj.bias.shape}, 范围=[{proj.bias.min().item():.6f}, {proj.bias.max().item():.6f}]")
            else:
                print("  偏置: None")
        
        print("-" * 50)
    
    # 分析位置编码
    if hasattr(jittor_model, 'pos_embed'):
        print("🔍 位置编码分析:")
        pos_embed = jittor_model.pos_embed
        print(f"  形状: {pos_embed.shape}")
        print(f"  统计: 均值={pos_embed.mean().item():.6f}, 标准差={pos_embed.std().item():.6f}")
        print("-" * 50)
    
    # 分析类别token
    if hasattr(jittor_model, 'cls_token'):
        print("🔍 CLS Token 分析:")
        cls_token = jittor_model.cls_token
        print(f"  形状: {cls_token.shape}")
        print(f"  统计: 均值={cls_token.mean().item():.6f}, 标准差={cls_token.std().item():.6f}")
        print("-" * 50)
    
    # 分析Transformer块
    if hasattr(jittor_model, 'blocks'):
        print("🔍 TRANSFORMER 块分析:")
        print(f"  总块数: {len(jittor_model.blocks)}")
        
        for i, block in enumerate(jittor_model.blocks[:2]):  # 只显示前2个块
            print(f"  Block {i}:")
            
            # 注意力层
            if hasattr(block, 'attn'):
                attn = block.attn
                if hasattr(attn, 'qkv'):
                    print(f"    QKV权重: {attn.qkv.weight.shape}")
                if hasattr(attn, 'proj'):
                    print(f"    输出投影: {attn.proj.weight.shape}")
            
            # MLP层
            if hasattr(block, 'mlp'):
                mlp = block.mlp
                if hasattr(mlp, 'fc1'):
                    print(f"    MLP FC1: {mlp.fc1.weight.shape}")
                if hasattr(mlp, 'fc2'):
                    print(f"    MLP FC2: {mlp.fc2.weight.shape}")
        
        if len(jittor_model.blocks) > 2:
            print(f"    ... 还有 {len(jittor_model.blocks) - 2} 个块")
        
        print("-" * 50)
    
    # 分析分类头
    if hasattr(jittor_model, 'head'):
        print("🔍 分类头分析:")
        head = jittor_model.head
        if hasattr(head, 'weight'):
            print(f"  权重形状: {head.weight.shape}")
            print(f"  统计: 均值={head.weight.mean().item():.6f}, 标准差={head.weight.std().item():.6f}")
        print("-" * 50)
    
    print("="*80)
def compare_weights_after_conversion(pytorch_model, jittor_model):
    """转换后权重对比分析"""
    
    print("="*80)
    print("PYTORCH vs JITTOR 权重对比")
    print("="*80)
    
    # 获取PyTorch状态字典
    pt_state_dict = pytorch_model.state_dict()
    
    # 对比关键层
    comparison_results = []
    
    for name, jt_param in jittor_model.named_parameters():
        if name in pt_state_dict:
            pt_param = pt_state_dict[name]
            
            # 转换为numpy进行对比
            pt_numpy = pt_param.detach().cpu().numpy()
            jt_numpy = jt_param.numpy()
            
            # 计算差异
            if pt_numpy.shape == jt_numpy.shape:
                diff = np.abs(pt_numpy - jt_numpy)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                result = {
                    'name': name,
                    'shape': pt_numpy.shape,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'match': max_diff < 1e-6
                }
            else:
                result = {
                    'name': name,
                    'shape_pt': pt_numpy.shape,
                    'shape_jt': jt_numpy.shape,
                    'match': False,
                    'error': 'Shape mismatch'
                }
            
            comparison_results.append(result)
    
    # 打印对比结果
    for result in comparison_results:
        print(f"参数: {result['name']}")
        
        if 'error' in result:
            print(f"  ❌ {result['error']}")
            print(f"  PyTorch形状: {result.get('shape_pt', 'N/A')}")
            print(f"  Jittor形状: {result.get('shape_jt', 'N/A')}")
        else:
            status = "✅ 匹配" if result['match'] else "⚠️  有差异"
            print(f"  {status}")
            print(f"  形状: {result['shape']}")
            print(f"  最大差异: {result['max_diff']:.2e}")
            print(f"  平均差异: {result['mean_diff']:.2e}")
        
        print("-" * 50)
    
    # 统计信息
    total_params = len(comparison_results)
    matched_params = sum(1 for r in comparison_results if r.get('match', False))
    
    print(f"总参数数: {total_params}")
    print(f"匹配参数: {matched_params}")
    print(f"匹配率: {matched_params/total_params*100:.1f}%")
    print("="*80)
import tempfile
import numpy as np
import torch
import jittor as jt
def convert_pytorch_to_jittor_with_analysis(pytorch_model, jittor_model):
    """转换并分析模型权重"""
    
    print("开始转换PyTorch模型到Jittor...")
    
    # 转换前分析
    # print("\n📊 转换前Jittor模型状态:")
    # print_jittor_model_weights(jittor_model)
    
    # 执行转换
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(pytorch_model.state_dict(), tmp.name)
        pt_state_dict = torch.load(tmp.name, map_location='cpu')
        
        successful_loads = 0
        failed_loads = 0
        
        for name, param in jittor_model.named_parameters():
            if name in pt_state_dict:
                try:
                    tensor = pt_state_dict[name].numpy()
                    param.assign(jt.array(tensor))
                    successful_loads += 1
                    # print(f"✅ 成功加载: {name} {tensor.shape}")
                except Exception as e:
                    failed_loads += 1
                    print(f"❌ 加载失败: {name} - {e}")
            else:
                failed_loads += 1
                print(f"⚠️  未找到参数: {name}")
        
        print(f"\n加载统计: 成功 {successful_loads}, 失败 {failed_loads}")
    
    # # 转换后分析
    # print("\n📊 转换后详细分析:")
    # print_model_structure_and_weights(jittor_model)
    
    # print("\n📊 关键层分析:")
    # analyze_specific_layers(jittor_model)
    
    # print("\n📊 权重对比分析:")
    # compare_weights_after_conversion(pytorch_model, jittor_model)
    
    return jittor_model

