# Contributing

We welcome contributions to the project. Please follow the guidelines below.

## Development environment

We use [pixi](https://pixi.sh/latest/) to manage the development environment.

```shell
uv venv
uv pip install -e ".[dev]"

source .venv/bin/activate
```

### Testing

Use pytest to run the unit checks:

```bash
pytest .
```

### Check all files

**Recommended before creating a commit**: to run all checks against all files,
run the following command:

```bash
pre-commit run --all-files
```
