from fastapi import FastAPI
import uvicorn

# We must import from the local module, which is now in server/
from .environment import DriverSafetyEnv, Action

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Driver Safety OpenEnv is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# OpenEnv runtime wrapper instance
server_env = DriverSafetyEnv()

@app.post("/reset")
def reset_endpoint():
    return server_env.reset()

@app.post("/step")
def step_endpoint(action: Action):
    return server_env.step(action)

@app.get("/state")
def state_endpoint():
    return server_env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
