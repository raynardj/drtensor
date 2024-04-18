# Doctor Tensor 🩺

```
pip install drtensor
```

> Debug tool to help pytorch modeling

## 📦 Installation
We assume you have the pytorch already installed. Or this library doesn't serve any purpose.

```bash
pip install drtensor
```

## 🚀 Usage
### With clause
```python
from drtensor.doctor import DrTensor


model = reset_model()

dr_tensor = DrTensor(resnet=model)

with dr_tensor:
    y_ = model(torch.randn(1, 3, 224, 224))

# check the logs
print(dr_tensor.logs[-5:])
```