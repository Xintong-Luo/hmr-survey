# 评测示例

[返回综述首页](../README.md)

> **从哪里开始？** 想复核论文结果，选 **HaMeR**。想先跑通评分，选 **FreeMoCap**，安装 NumPy 后即可用随附的六段轨迹离线运行。

## 一、选择示例

### 1. [HaMeR · 手部网格](freihand_table7/README.md)

在 FreiHAND 全部 **3,960 张测试图**上复核手部顶点误差。CPU 评分需要测试集和 MANO 右手模型。从图像重新推理还需要 CUDA GPU 和 HaMeR 权重。

### 2. [GLAMR · 单段骨盆轨迹](emdb_glamr_rte/README.md)

用一段 **2,240 帧**的室外行走序列，查看预测轨迹在哪些帧偏离参考轨迹。CPU 和 NumPy 即可评分，脚本会下载 EMDB 官方样本。

### 3. [FreeMoCap · 多段骨盆轨迹](freemocap_multivideo_rte/README.md)

对 **3 名参与者的 6 段跑台步行**逐段评分，再汇总 42,243 帧的结果。六段轨迹都已随仓库提供，下面的快速入门使用这个示例。

## 二、第一次运行

### 1. 下载仓库

```bash
git clone https://github.com/Xintong-Luo/hmr-survey.git
cd hmr-survey
```

如果从 GitHub 下载了 ZIP，解压后在含有 `evaluation/` 的目录打开终端即可。三个示例的运行说明都从这个位置开始。

### 2. 用随附轨迹做一次评分

以下命令适用于 Linux 或 macOS。进入 FreeMoCap 示例目录，安装 NumPy 后即可离线运行。

```bash
cd evaluation/freemocap_multivideo_rte
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
mkdir -p runs
python evaluate.py --csv runs/results.csv
```

终端将依次显示六段录制的结果。六段等权平均的根轨迹误差应约为 **0.02780%**，按计分帧数加权应约为 **0.02725%**。可把新生成的 `runs/results.csv` 与[随附的六段参考结果](freemocap_multivideo_rte/results.csv)逐行核对。

## 三、哪些结果可以比较

- **手部网格误差。** HaMeR 的 PA-MPVPE 以毫米计，衡量每张图中手部网格对齐后的顶点误差。完整测试集实测 **5.7056 mm**，保留一位小数为 **5.7 mm**，与综述表 7 使用同一数据和指标。[HaMeR 参考结果](freihand_table7/result.json)
- **骨盆轨迹误差。** RTE 以百分比计，把对齐后的平均位置误差除以该段参考骨盆走过的总长度。GLAMR 的 **19.19%** 来自一段室外行走，FreeMoCap 的 **0.02780%** 来自六段跑台录制的等权平均。数据、采集场景和轨迹长度均不同，两个数值各自对应一个评分示例，不能用于方法排名。[GLAMR 参考结果](emdb_glamr_rte/result.json) · [FreeMoCap 逐段结果](freemocap_multivideo_rte/results.csv)
