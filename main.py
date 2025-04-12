import json
import os
import io
import time
from typing import Set
from PIL import Image

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# import firebase_admin
from pydantic import BaseModel

# from firebase_admin import credentials, db
from scalar_fastapi import get_scalar_api_reference
from ultralytics import YOLO

load_dotenv()

app = FastAPI(title="Body Tracker API", description="API for tracking workout")

model = YOLO("yolo11n-pose.pt")

# cred = credentials.Certificate(
#     os.getenv("FIREBASE_SERVICE_ACCOUNT")
# )  # Get this from firebase console
# firebase_admin.initialize_app(cred, {"databaseURL": os.getenv("FIREBASE_DATABASE_URL")})


def save_data(user_id, workout_type, rep_count, timestamp):
    print(
        f"Saving data for user {user_id} in workout {workout_type} with {rep_count} reps at {timestamp}"
    )
    # ref = db.reference(f"/users/{user_id}/workouts/{workout_type}/{timestamp}")
    # ref.set({"reps": rep_count, "timestamp": timestamp})


def count_squats(keypoints, confidence, orig_shape) -> bool:
    """
    Given a set of keypoints, detects whether the user is performing a squat or not.
    """
    # Define the keypoint indices for the knees and hips
    left_knee_idx = 13
    right_knee_idx = 14
    left_hip_idx = 11
    right_hip_idx = 12
    left_ankle_idx = 15
    right_ankle_idx = 16

    threshold = orig_shape[1] / 100 * 7

    # Get the coordinates of the knees and hips
    if (
        confidence[left_knee_idx] < 0.5
        or confidence[right_knee_idx] < 0.5
        or confidence[left_hip_idx] < 0.5
        or confidence[right_hip_idx] < 0.5
    ):
        return False
    left_knee = keypoints[left_knee_idx]
    right_knee = keypoints[right_knee_idx]
    left_hip = keypoints[left_hip_idx]
    right_hip = keypoints[right_hip_idx]
    left_ankle = keypoints[left_ankle_idx]
    right_ankle = keypoints[right_ankle_idx]

    # # calculate the hip angle
    # hip_angle = (
    #     np.arccos(
    #         np.dot(
    #             (left_knee - left_hip) / np.linalg.norm(left_knee - left_hip),
    #             (left_ankle - left_hip) / np.linalg.norm(left_ankle - left_hip),
    #         )
    #     )
    #     + np.arccos(
    #         np.dot(
    #             (right_knee - right_hip) / np.linalg.norm(right_knee - right_hip),
    #             (right_ankle - right_hip) / np.linalg.norm(right_ankle - right_hip),
    #         )
    #     )
    # ) / 2

    # # calculate the knee angle
    # knee_angle = (
    #     np.arccos(
    #         np.dot(
    #             (left_knee - left_hip) / np.linalg.norm(left_knee - left_hip),
    #             (left_ankle - left_knee) / np.linalg.norm(left_ankle - left_knee),
    #         )
    #     )
    #     + np.arccos(
    #         np.dot(
    #             (right_knee - right_hip) / np.linalg.norm(right_knee - right_hip),
    #             (right_ankle - right_knee) / np.linalg.norm(right_ankle - right_knee),
    #         )
    #     )
    # ) / 2

    # if knee_angle <= 115 and hip_angle > 160:
    #     print("DOWN")
    # elif knee_angle > 140 and hip_angle > 160:
    #     print("UP")

    return (
        abs(left_hip[1] - left_knee[1]) < threshold
        and abs(right_hip[1] - right_knee[1]) < threshold
    )


class ConnectionInfo(BaseModel):
    frame_count: int
    squat_count: int


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: dict[WebSocket, ConnectionInfo] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_info[websocket] = ConnectionInfo(frame_count=0, squat_count=0)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        del self.connection_info[websocket]

    async def send_personal_json(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)

        while True:
            try:
                if websocket.state == 0:  # 0 is CONNECTING state
                    continue
                if websocket.state == 2:  # 2 is DISCONNECTED state
                    print("WebSocket is not connected")
                    break

                # Wait for frame
                frame = await websocket.receive_bytes()
                if frame is None:  # Connection closed
                    print("Received None frame, connection closed")
                    break

                manager.connection_info[websocket].frame_count += 1

                # image = cv2.imdecode(
                #     np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
                # )
                # if image is None:
                #     print("Failed to decode frame")
                #     continue

                image = Image.open(io.BytesIO(frame))
                results = model(image, verbose=False)

                if (
                    len(results) > 0
                    and results[0] is not None
                    and results[0].names[0] == "person"
                    and results[0].keypoints is not None
                    and results[0].keypoints.conf is not None
                    and results[0].keypoints.orig_shape is not None
                ):
                    keypoints = results[0].keypoints.xy[0]

                    # print("keypoints", results[0].keypoints.conf)
                    if count_squats(
                        keypoints,
                        results[0].keypoints.conf[0],
                        results[0].keypoints.orig_shape,
                    ):
                        manager.connection_info[websocket].squat_count += 1

                    keypoints_data = {
                        "keypoints": keypoints.tolist(),
                        "confidence": results[0].keypoints.conf[0].tolist(),
                        "orig_shape": results[0].keypoints.orig_shape,
                        "squat_count": manager.connection_info[websocket].squat_count,
                    }

                    await manager.send_personal_json(keypoints_data, websocket)

            except Exception as e:
                import traceback
                error_msg = f"Error processing frame: {str(e)}\n"
                error_msg += f"Traceback:\n{traceback.format_exc()}"
                print(error_msg)
                if isinstance(e, (RuntimeError, ConnectionError)):
                    print("Connection error detected, breaking loop")
                    break
                continue
    except Exception as e:
        print(f"WebSocket connection error: {str(e)}")
    finally:
        try:
            if websocket.state == 1:
                print("Closing websocket connection")
                await websocket.close()
            manager.disconnect(websocket)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


@app.get("/index", include_in_schema=False, response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        return f.read()


@app.get("/ui", include_in_schema=False, response_class=HTMLResponse)
async def get_docs():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
