# HaMeR · 手部网格评测

[评测目录](../README.md) · [评分脚本](evaluate.py) · [参考结果](result.json) · [综述首页](../../README.md)

> HaMeR 在 FreiHAND v2 全部 **3,960 张测试图像**上的手部网格误差为 **5.7 mm**（PA-MPVPE，保留一位小数）。复核这个数值时，可以直接用 CPU 给随附的预测评分。从测试图像重新生成预测时，使用 GPU 路线。

## 一、数据与准备

使用 Linux x86-64 和 Python 3.10。从本仓库根目录执行一次下列命令，后续步骤均在该示例目录内运行。

```bash
cd evaluation/freihand_table7
mkdir -p data models runs
```

| 需要的文件 | 下载和放置位置 |
| --- | --- |
| **FreiHAND v2 测试集**（约 724 MiB） | 下载 **Evaluation set with annotations**，保存为 `data/FreiHAND_pub_v2_eval.zip`。CPU 评分直接读取 ZIP 内的参考网格。[官方下载](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip) · [数据集说明](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) |
| **MANO v1.2 右手模型** | 登录官网，在 **Download → Models & Code** 下载 `mano_v1_2.zip`，在解压后的 `mano_v1_2/models/` 中找到 `MANO_RIGHT.pkl`，复制到本示例的 `models/` 目录。[官方下载页](https://mano.is.tue.mpg.de/download.php) |
| **HaMeR 逐图预测**（约 2.2 MiB） | 已随仓库提供。CPU 评分读取每张图的手部旋转与形状参数，再用 MANO 生成网格。[预测文件](predictions/hamer_freihand_params.npz) |

## 二、运行评测

### 1. CPU · 给已有预测评分

在示例目录中创建 CPU 环境。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r requirements-score.txt
```

运行评分脚本。终端会打印结果，同时保存一份到 `runs/from_saved_predictions.json`。

```bash
OMP_NUM_THREADS=2 python evaluate.py \
  --reference data/FreiHAND_pub_v2_eval.zip \
  --parameters predictions/hamer_freihand_params.npz \
  --mano models/MANO_RIGHT.pkl \
  --output runs/from_saved_predictions.json
```

### 2. GPU · 从测试图像重新推理

准备好第一节的测试集和 MANO 模型后，在支持 CUDA 的 NVIDIA GPU 上运行。本例使用 Tesla T4（16 GB），每批处理 16 张图像。

<details>
<summary><strong>展开 GPU 环境、官方权重下载和推理命令</strong></summary>

**安装独立的推理环境**（位于本示例目录）。

```bash
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install --no-build-isolation -r requirements.txt
```

**取得 HaMeR 代码、权重和测试清单。** 下列命令使用本次复测的代码和权重版本。测试清单 `freihand_val.npz` 给出图像顺序与裁剪框。[官方评测说明](https://github.com/geopavlakos/hamer#evaluation) · [官方权重与配置](https://huggingface.co/spaces/geopavlakos/HaMeR/tree/aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a/_DATA/hamer_ckpts)

```bash
mkdir -p vendor
git clone https://github.com/geopavlakos/hamer.git vendor/hamer
git -C vendor/hamer checkout 3a01849f4148352e9260b69bf28b65d1671a4905

weights_url='https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a/_DATA/hamer_ckpts'
curl -fL --retry 3 "$weights_url/checkpoints/hamer.ckpt" -o models/hamer.ckpt
curl -fL --retry 3 "$weights_url/model_config.yaml" -o models/model_config.yaml
curl -fL --retry 3 'https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh&dl=1' -o data/hamer_evaluation_data.tar.gz

tar -xzf data/hamer_evaluation_data.tar.gz -C data hamer_evaluation_data/freihand_val.npz
unzip -q -n data/FreiHAND_pub_v2_eval.zip 'evaluation/rgb/*.jpg' -d data
```

**生成逐图预测并评分。** 第一个命令把图像预测写入 `runs/hamer_freihand_params.npz`，第二个命令读取这份新预测并计算误差。

```bash
python predict_parameters.py \
  --repo vendor/hamer \
  --checkpoint models/hamer.ckpt --config models/model_config.yaml \
  --meta data/hamer_evaluation_data/freihand_val.npz --rgb data/evaluation/rgb \
  --output runs/hamer_freihand_params.npz

OMP_NUM_THREADS=2 python evaluate.py \
  --reference data/FreiHAND_pub_v2_eval.zip \
  --parameters runs/hamer_freihand_params.npz --mano models/MANO_RIGHT.pkl \
  --output runs/from_images.json
```

</details>

<details>
<summary>已有官方评测导出的 freihand-val.json？直接给顶点预测评分</summary>

将 HaMeR 官方 `eval.py --dataset FREIHAND-VAL` 导出的 `freihand-val.json` 放在本示例目录，激活上述任一 Python 环境后运行。

```bash
python evaluate.py --reference data/FreiHAND_pub_v2_eval.zip \
  --submission freihand-val.json --output runs/from_submission.json
```

</details>

## 三、结果解读

先看输出中的 `images` 和 `vertices_per_image`，完整测试应分别为 **3960** 和 **778**。再读 `pa_mpvpe_mm`，它是对齐后的平均顶点误差，单位为毫米，越小越好。评分时，每张图的预测网格先与参考网格独立对齐，再计算对应顶点的平均距离，最后对全部图像取平均。[FreiHAND 官方评分实现](https://github.com/lmb-freiburg/freihand/blob/master/eval.py)

| 结果来源 | PA-MPVPE（mm） | 保留一位小数（mm） |
| --- | ---: | ---: |
| CPU · 对随附预测评分 | 5.705628416 | **5.7** |
| GPU · 从原始图像重新推理，再评分 | 5.705628359 | **5.7** |
| HaMeR 原论文与综述表 7 | 5.7 | **5.7** |

两条路线均在 Linux、Python 3.10.22 和 PyTorch 2.6.0 下完成。仓库中的 `result.json` 保存的是 CPU 对随附预测的结果；用 FreiHAND 原始评分函数独立复算，得到的误差同为 5.7056 mm。[参考输出](result.json) · [HaMeR 原论文结果](https://arxiv.org/pdf/2312.05251#page=5)

## 四、引用与资源

- **HaMeR 方法**　Pavlakos 等，*Reconstructing Hands in 3D with Transformers*，CVPR 2024。[论文](https://arxiv.org/abs/2312.05251)
- **FreiHAND 数据集**　Zimmermann 等，*FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images*，ICCV 2019。[数据集与引用信息](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- **MANO 手部模型**　Romero、Tzionas、Black，*Embodied Hands: Modeling and Capturing Hands and Bodies Together*，SIGGRAPH Asia 2017。[项目与引用信息](https://mano.is.tue.mpg.de/) · [模型许可](https://mano.is.tue.mpg.de/license.html)
