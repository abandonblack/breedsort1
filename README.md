# 猫狗品种精准识别平台

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示
- 对比 **SE-ResNet34** 与 **ResNet34（无注意力）**

## 模型与数据集

- **SE-ResNet34**（含 SE 通道注意力，不调用 `torchvision.models`）
- **ResNet34**（无注意力机制）
- **Oxford-IIIT Pet Dataset**（通过 `torchvision.datasets.OxfordIIITPet` 加载）

## 1. 安装依赖

```bash
pip install torch torchvision fastapi uvicorn pillow jinja2 python-multipart matplotlib
```

## 2. 分别训练两个网络

### 2.1 训练 SE-ResNet34

```bash
python -m app.train_seresnet34 --data-dir data/oxford_iiit_pet --epochs 40 --batch-size 32
```

默认输出：

- `artifacts/seresnet34.pth`
- `artifacts/seresnet34_history.json`

### 2.2 训练 ResNet34（无注意力）

```bash
python -m app.train_resnet34 --data-dir data/oxford_iiit_pet --epochs 40 --batch-size 32
```

默认输出：

- `artifacts/resnet34_plain.pth`
- `artifacts/resnet34_plain_history.json`

## 3. 绘制训练对比图 + 官方 test 指标

```bash
python -m app.compare_models --data-dir data/oxford_iiit_pet
```

输出：

- `artifacts/compare_train_loss.png`（两个网络 train loss 对比）
- `artifacts/compare_val_acc.png`（两个网络 val acc 对比）
- `artifacts/compare_test_topk.png`（官方 test 上 top1/top3 柱状图）
- `artifacts/compare_report.json`（top1/top3 数值）

## 4. 启动服务

```bash
python -m app.main
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看数据集清单（当前仅 Oxford-IIIT）
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
