import uvicorn
from main import app
from src.api.config import Config

if __name__ == "__main__":
    print(f"Server: http://{Config.HOST}:{Config.PORT}")
    print(f"API Docs: http://{Config.HOST}:{Config.PORT}/docs\n")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
