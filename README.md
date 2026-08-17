# safetensors-viewer (`stview`)

一个轻量的 Hugging Face safetensors 权重查看器。它只读取文件头，不会把数 GB 的 tensor 数据加载进内存。

底层使用官方 `safetensors` 库做格式验证，不依赖 PyTorch。

支持以下输入：

- 单个 `.safetensors` 文件
- Hugging Face `*.safetensors.index.json`
- 包含单文件或分片权重的模型目录

## 安装

需要 Python 3.10 或更高版本。

### 使用 uv（推荐）

```bash
git clone https://github.com/BangBOOM/safetensors-viewer.git
cd safetensors-viewer
uv sync
uv run stview --help
```

不想 clone 也可以直接运行或安装为全局命令：

```bash
uvx --from git+https://github.com/BangBOOM/safetensors-viewer stview --help
uv tool install git+https://github.com/BangBOOM/safetensors-viewer
```

### 使用 pip

```bash
git clone https://github.com/BangBOOM/safetensors-viewer.git
cd safetensors-viewer
python -m pip install -e .
stview --help
```

也可以直接从 Git 安装：

```bash
python -m pip install git+https://github.com/BangBOOM/safetensors-viewer.git
```

安装完成后 `stview` 就是一个普通可执行命令，下面示例里的 `uv run stview` 都可以直接写成 `stview`。

## 命令

### 模型概览

```bash
uv run stview info /path/to/model
uv run stview info /path/to/model/model.safetensors.index.json --format json
```

显示 tensor 数、参数量、逻辑数据大小、dtype/rank 分布，以及每个分片的信息。

### Tensor 列表

```bash
uv run stview list /path/to/model
uv run stview list /path/to/model --match 'blocks.0.*'
uv run stview list /path/to/model --regex 'blocks\\.(0|1)\\.' --ndim 2
uv run stview list /path/to/model --dtype BF16 --sort size --desc --limit 20
uv run stview list /path/to/model --format json
uv run stview list /path/to/model --format csv
```

表格包括名称、shape、dtype、参数量、字节数和所属分片。

### 查看单个 Tensor

```bash
uv run stview show /path/to/model blocks.0.attn.to_q.weight
```

显示完整 shape、rank、dtype、参数量、数据偏移和分片路径。

### 模块树

```bash
uv run stview tree /path/to/model --depth 3
```

按照 tensor 名称中的 `.` 构造模型层级树。省略 `--depth` 会显示完整结构。

### 完整性检查

```bash
uv run stview verify /path/to/model
```

检查：

- Hugging Face 索引与实际分片是否一致
- tensor 是否缺失、重复或未被索引记录
- shape、dtype 与数据长度是否一致
- data offsets 是否越界或重叠
- 索引中的 `total_size` 是否正确
- 官方 `safetensors` 库能否打开每个分片

发现错误时命令退出码为 `1`，输入或参数错误时退出码为 `2`。

### Web Viewer

```bash
uv run stview serve /path/to/model
```

然后访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。也可以自动打开默认浏览器：

```bash
uv run stview serve /path/to/model --open
uv run stview serve /path/to/model --host 127.0.0.1 --port 8765
```

Web 界面提供：

- 模型参数量、数据大小、dtype 和 rank 分布
- 按模块聚合的模型结构概览
- Tensor 名称搜索、dtype/rank/分片筛选、排序与分页
- Tensor shape、dtype、offset 和所属分片详情
- 分片占比与 metadata
- 一键运行完整性检查
- 位于 `/api/docs` 的交互式 REST API 文档

模型 header 仅在服务启动时读取一次，搜索和分页不会重复扫描权重文件。服务默认只监听本机 `127.0.0.1`；如果通过 `--host 0.0.0.0` 暴露到局域网，请确认网络环境可信。

## 性能特点

`info`、`list`、`show`、`tree` 和 Web Viewer 只读取每个 safetensors 文件开头的 JSON header。模型是 1 GB 还是 100 GB，内存占用主要取决于 tensor 数量，而不是权重数据大小。

## 开发

```bash
uv sync          # 安装运行依赖 + dev 依赖组（pytest、httpx）
uv run pytest
uv build         # 构建 sdist 和 wheel 到 dist/
```

不用 uv 的话，dev 依赖是 PEP 735 的 dependency group，需要手动装：

```bash
python -m pip install -e .
python -m pip install pytest httpx
pytest
```

## License

[MIT](LICENSE)
