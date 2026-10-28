default:
    @just --list

lab port="8888":
    nohup uv run --no-sync jupyter lab --no-browser --port={{port}} > jupyter.log 2>&1 &

token:
    grep -m1 token jupyter.log