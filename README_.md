![.](/image/task1.png)

![.](/image/task2.png)

![Alt Text](/image/task3.png)

![Alt Text](/image/task4.png)

![Alt Text](/image/task5.png)

![Alt Text](/image/task6.png)

![Alt Text](/image/task7.png)

![Alt Text](/image/task8.png)

![Alt Text](/image/task9.png)

![Alt Text](/image/task10.png)

一共100个类别，分为10个任务进行训练

## Pytorch框架下，CIFAR00，acc@1（灰色标注行）

![Alt Text](/image/torchCifar100.png)

## Jittor框架下acc@1

![Alt Text](/image/jittorCifar100.png)

## Pytorch框架下，5个数据集，acc@1（灰色标注行）

![Alt Text](/image/torchFive.png)

## Jittor框架下

![Alt Text](/image/jittorFive.png)

## 训练终端命令

### 

#### 选择five_datasets_l2p, --output_dir ./output_FashionMNIST即训练FashionMNIST数据集。更改参数 --output_dir./output_MNIST即为训练MNIST数据集。剩下数据集依次类推

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py five_datasets_l2p --model vit_base_patch16_224 --batch-size 8 --data-path ./local_datasets/ --output_dir ./output_FashionMNIST

### 训练以及测试CIFAR100

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 8 --data-path ./local_datasets/ --output_dir ./output

## 准备数据集，只用将，dataset.py文件下传入的参数设置为Ture就会自动下载

![Alt Text](/image/python.png)

### **完整 Conda 环境搭建指南（支持 CUDA 的 Jittor + PyTorch）**

---

## **1. 创建 Conda 环境（Python 3.7.12）**

```bash
conda create -n jittor python=3.7.12 -y
conda activate jittor  # 进入环境
```

✅ **检查 Python 版本：**

```bash
python --version  # 确保是 Python 3.7.12
```

---

## **2. 检查 CUDA 版本（关键！）**

📌 **Jittor (`>=1.3.9.14`)** 支持：

- **CUDA 11.1**（推荐）
- PyTorch (`1.13.1`) 支持 **CUDA 11.6~11.7**

🔹 **如果你的 CUDA 版本是 11.1~11.7（大多数情况）：**

```bash
nvidia-smi  # 查看 CUDA 版本
nvcc --version  # 如果安装了 NVCC，显示 CUDA 编译器版本
```

- 如果版本不一致，建议先 **安装 CUDA 11.7**（兼容 PyTorch 1.13.1）：
  
  ```bash
  conda install -c nvidia cuda-toolkit=11.7 -y
  ```

---

## **3. 安装 PyTorch（GPU 版本）**

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

✅ **测试 PyTorch CUDA 支持：**

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

📌 **如果 `torch.cuda.is_available()` 返回 `True`，说明 GPU 支持正常。**

---

## **4. 安装 Jittor (GPU 版本)**

```bash
pip install jittor==1.3.9.14
```

✅ **检查 Jittor GPU 支持：**

```python
python -c "import jittor as jt; print(f'Jittor: {jt.__version__}, GPU: {jt.has_cuda}')"
```

📌 **如果 `jt.has_cuda` 返回 `True`，说明 Jittor 也能访问 GPU。**

---

## **5. 安装剩余依赖 (+timm, matplotlib, tqdm 等)**

```bash
pip install timm==0.6.7 pillow==9.2.0 matplotlib==3.5.3 torchprofile==0.0.4 scipy==1.9.3 tqdm==4.67.1
```

📌 验证全部安装正确：

```bash
pip list | grep -E "jittor|torch|timm|pillow|matplotlib|torchprofile|scipy|tqdm"
```

✅ **示例输出：**

```
jittor              1.3.9.14
torch               1.13.1+cu117
torchvision         0.14.1+cu117
timm                0.6.7
pillow              9.2.0
matplotlib          3.5.3
torchprofile        0.0.4
scipy               1.9.3
tqdm                4.67.1
```
