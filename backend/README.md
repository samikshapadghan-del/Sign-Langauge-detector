# Backend

Run the project from the repository root with `run.ps1`.

The production ASGI entrypoint is:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 5000
```

See the root `README.md` for setup, training, deployment, and accuracy limitations.
