from pydantic import BaseModel, Field

class DriverSafetyAction(BaseModel):
    action_type: str = Field(..., description="Action to take: BEEP, VOICE, ALARM, BLOCK_IGNITION, NONE")

class DriverSafetyObservation(BaseModel):
    ear: float = Field(..., description="Eye Aspect Ratio")
    drowsiness_state: str = Field(..., description="Current drowsiness state")
    drunk_status: str = Field(..., description="Current drunk status")

class DriverSafetyState(BaseModel):
    ear: float = Field(..., description="Eye Aspect Ratio")
    drowsiness_state: str = Field(..., description="Current drowsiness state")
    ignore_count: int = Field(..., description="Number of ignored alerts")
    sway_variance: float = Field(..., description="Sway variance metric")
    eye_asymmetry: float = Field(..., description="Eye asymmetry metric")
    drunk_status: str = Field(..., description="Current drunk status")
