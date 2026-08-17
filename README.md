# safetensors-viewer (`stview`)

A lightweight inspector for Hugging Face safetensors weights. It reads only the
file headers, so multi-gigabyte tensor data is never loaded into memory.

Format validation is delegated to the official `safetensors` library. PyTorch is
not required.

Accepted inputs:

- a single `.safetensors` file
- a Hugging Face `*.safetensors.index.json`
- a model directory containing single-file or sharded weights

## Installation

Requires Python 3.10 or newer.

### With uv (recommended)

```bash
git clone https://github.com/BangBOOM/safetensors-viewer.git
cd safetensors-viewer
uv sync
uv run stview --help
```

You can also run it without cloning, or install it as a global command:

```bash
uvx --from git+https://github.com/BangBOOM/safetensors-viewer stview --help
uv tool install git+https://github.com/BangBOOM/safetensors-viewer
```

### With pip

```bash
git clone https://github.com/BangBOOM/safetensors-viewer.git
cd safetensors-viewer
python -m pip install -e .
stview --help
```

Or install straight from Git:

```bash
python -m pip install git+https://github.com/BangBOOM/safetensors-viewer.git
```

Once installed, `stview` is a regular executable on your `PATH`. Every
`uv run stview` in the examples below can be shortened to just `stview`.

## Commands

### Model overview

```bash
uv run stview info /path/to/model
uv run stview info /path/to/model/model.safetensors.index.json --format json
```

Reports tensor count, parameter count, logical data size, dtype and rank
distributions, and a per-shard breakdown.

### Listing tensors

```bash
uv run stview list /path/to/model
uv run stview list /path/to/model --match 'blocks.0.*'
uv run stview list /path/to/model --regex 'blocks\.(0|1)\.' --ndim 2
uv run stview list /path/to/model --dtype BF16 --sort size --desc --limit 20
uv run stview list /path/to/model --shard 'model-00001-*'
uv run stview list /path/to/model --format json
uv run stview list /path/to/model --format csv
```

The table shows name, shape, dtype, parameter count, byte size, and owning
shard. Sort keys: `name`, `size`, `params`, `dtype`, `rank`.

### Inspecting one tensor

```bash
uv run stview show /path/to/model blocks.0.attn.to_q.weight
```

Prints the full shape, rank, dtype, parameter count, data offsets, and the path
of the shard holding it.

### Module tree

```bash
uv run stview tree /path/to/model --depth 3
```

Builds a module hierarchy by splitting tensor names on `.`. Omit `--depth` to
show the complete structure.

### Integrity check

```bash
uv run stview verify /path/to/model
```

Checks that:

- the Hugging Face index agrees with the shards actually present
- no tensor is missing, duplicated, or absent from the index
- shape, dtype, and data length are mutually consistent
- data offsets stay in bounds and never overlap
- `total_size` in the index is correct
- every shard can be opened by the official `safetensors` library

The command exits `1` when problems are found, and `2` on bad input or
arguments.

### Web viewer

```bash
uv run stview serve /path/to/model
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). You can also let it
open your default browser, or pick a different address:

```bash
uv run stview serve /path/to/model --open
uv run stview serve /path/to/model --host 127.0.0.1 --port 8765
```

The web UI offers:

- parameter count, data size, and dtype / rank distributions
- a structural overview aggregated by module
- search by tensor name, filters on dtype / rank / shard, sorting, and paging
- per-tensor shape, dtype, offsets, and owning shard
- shard size share and metadata
- one-click integrity check
- interactive REST API docs at `/api/docs`

Headers are read once at startup, so searching and paging never rescan the
weight files. The server binds to `127.0.0.1` by default; if you expose it with
`--host 0.0.0.0`, make sure the network is trusted.

## Performance

`info`, `list`, `show`, `tree`, and the web viewer only read the JSON header at
the start of each safetensors file. Whether the model is 1 GB or 100 GB, memory
use scales with the number of tensors, not with the size of the weight data.

## Development

```bash
uv sync          # runtime deps + the dev dependency group (pytest, httpx)
uv run pytest
uv build         # build sdist and wheel into dist/
```

Without uv, note that the dev dependencies live in a PEP 735 dependency group
and have to be installed separately:

```bash
python -m pip install -e .
python -m pip install pytest httpx
pytest
```

## License

[MIT](LICENSE)
