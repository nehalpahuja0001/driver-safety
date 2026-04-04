from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "reset":
                state = server_env.reset()
                await websocket.send_json({"type": "reset", "data": state.model_dump()})
                
            elif msg_type == "step":
                action_val = data.get("action", "NONE")
                action_obj = Action(action_type=action_val)
                result = server_env.step(action_obj)
                await websocket.send_json({"type": "step", "data": result.model_dump()})
                
            else:
                await websocket.send_json({"type": "error", "error": "Unknown message type"})
                
    except WebSocketDisconnect:
        # Client gracefully closed connection
        pass
    except Exception as e:
        # Handle JSON parse errors or internal errors gracefully while keeping connection open if possible
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
