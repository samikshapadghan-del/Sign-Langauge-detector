import os

import uvicorn

from backend.app import app


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        reload=False,
    )
