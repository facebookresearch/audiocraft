# AudioCraft
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

AudioCraft 是一个用于音频生成深度学习研究的 PyTorch 算法库。AudioCraft 包含了用于生成高品质音频的两大前沿 AI 生成模型的完整推理与训练代码：AudioGen 与 MusicGen。


## 安装指南 (Installation)
AudioCraft 要求运行环境为 **Python 3.9** 与 **PyTorch 2.1.0**。可通过以下命令安装 AudioCraft：

```shell
# 建议先确保已安装 PyTorch，特别是在安装 xformers 之前。
# 如果您的环境中已经安装了对应版本的 PyTorch，请跳过此命令。
python -m pip install 'torch==2.1.0'
# 在安装软件包之前可能需要更新以下依赖项
python -m pip install setuptools wheel
# 随后根据需求选择以下安装方式之一
python -m pip install -U audiocraft  # 稳定发布版 (Stable release)
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # 前沿开发版 (Bleeding edge)
python -m pip install -e .  # 本地克隆安装（若需进行模型训练则必选此项）
python -m pip install -e '.[wm]'  # 若需训练音频水印（Watermarking）模型请选择此项
```

我们还强烈建议安装 `ffmpeg`，可以通过系统包管理器或 Anaconda 进行安装：
```bash
sudo apt-get install ffmpeg
# 或者使用 Anaconda / Miniconda 时
conda install "ffmpeg<5" -c conda-forge
```

## 模型生态 (Models)

目前，AudioCraft 提供了以下模型的训练代码与推理代码：
* [MusicGen](./docs/MUSICGEN.md)：前沿的可控文本到音乐生成模型（Text-to-Music）。
* [AudioGen](./docs/AUDIOGEN.md)：前沿的文本到音效生成模型（Text-to-Sound）。
* [EnCodec](./docs/ENCODEC.md)：高保真神经音频编解码器（Neural Audio Codec）。
* [Multi Band Diffusion](./docs/MBD.md)：兼容 EnCodec 的多频段扩散模型解码器。
* [MAGNeT](./docs/MAGNET.md)：用于文本到音乐与文本到音效的前沿非自回归（Non-autoregressive）生成模型。
* [AudioSeal](./docs/WATERMARKING.md)：前沿的鲁棒音频水印模型（Audio Watermarking）。
* [MusicGen Style](./docs/MUSICGEN_STYLE.md)：支持文本与风格共同控制的前沿音乐生成模型。
* [JASCO](./docs/JASCO.md)：支持和弦、旋律与鼓点多重条件控制的高质量文本到音乐生成模型。


## 训练代码 (Training code)

AudioCraft 包含了一整套用于音频深度学习研究的 PyTorch 组件，以及针对上述各模型的完整训练流水线（Pipelines）。
关于 AudioCraft 的核心架构设计理念以及如何构建自定义训练流水线的通用指引，请参阅 [AudioCraft 训练文档](./docs/TRAINING.md)。

若需复现现有学术研究成果或直接使用现有的训练流水线，请参阅各个特定模型的专属说明文档，其中提供了详细的配置文件指引、示例网格参数、针对特定任务的模型信息与常见问题解答。


## API 文档 (API documentation)

我们为 AudioCraft 提供了官方的 [API 参考文档](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/index.html)。


## 常见问题解答 (FAQ)

#### 是否提供训练代码？

是的！我们提供了 [EnCodec](./docs/ENCODEC.md)、[MusicGen](./docs/MUSICGEN.md)、[Multi Band Diffusion](./docs/MBD.md) 以及 [JASCO](./docs/JASCO.md) 的完整训练代码。

#### 模型权重缓存在哪里？

Hugging Face 会将模型权重存储在默认缓存目录中。针对 AudioCraft 相关模型，可以通过设置环境变量 `AUDIOCRAFT_CACHE_DIR` 来指定自定义存储路径。
若要更改其他 Hugging Face 模型的缓存位置，请参阅 [Hugging Face Transformers 缓存配置文档](https://huggingface.co/docs/transformers/installation#cache-setup)。
最后，如果您使用了依赖于 Demucs 的模型（例如 `musicgen-melody`）并希望更改 Demucs 的模型下载目录，请参考 [Torch Hub 官方文档](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)。


## 开源协议 (License)
* 本仓库中的**代码**基于 MIT 许可证发布，详见 [LICENSE 文件](LICENSE)。
* 本仓库中的**模型权重**基于知识共享署名-非商业性使用 4.0（CC-BY-NC 4.0）许可证发布，详见 [LICENSE_weights 文件](LICENSE_weights)。


## 引用说明 (Citation)

如需引用 AudioCraft 整体算法框架，请使用以下 BibTeX 条目：
```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

如需引用特定模型，请按照各模型独立文档中的说明进行引用，例如 [./docs/MUSICGEN.md](./docs/MUSICGEN.md)、[./docs/AUDIOGEN.md](./docs/AUDIOGEN.md) 等。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月16日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
