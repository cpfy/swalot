<h1 align="center">
<img src="assets/logo.svg" width="200">
</h1>

[![badge-pypi](https://img.shields.io/badge/pypi-v0.0.3-brightgreen)](https://pypi.org/project/swalot/) [![badge-python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/) [![badge-license](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/license/mit)

## Installation

Install the package using pip:

```bash
pip install swalot
```

If you are using pip mirror, you may get error logs such as "No matching distribution found.."
Try using pypi as the installation source:

```bash
pip install swalot -i https://pypi.python.org/simple
```



## Usage

Simply import and wrap training code:
```python
import swalot as sw

with sw.gpu():
    """
    use CUDA tensor calculation here as usual.
    All RAM will be protected automatically!
    e.g.
    """
    # a = torch.randn(1000, 1000, 600).cuda()
    # a = torch.randn(1000, 1000, 600).to("cuda:0")
    # a = torch.randn(1000, 1000, 600, device="cuda:0")
```

## 🗓️ Schedule

- [ ] Enable GPU protect mode with decorator
- [ ] Support multiple GPU environment