# Getting Started

## Install

### Using `uv`

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync
```

To include development tooling (tests + docs):

```bash
uv sync --extra dev
```

### Using `pip`

```bash
pip install -e ".[dev]"
```

## Serve docs locally

Once dependencies are installed:

```bash
mkdocs serve
```

Then open <http://127.0.0.1:8000>.
