import io, time
from typing import Set
import torch
import firebase_admin
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from firebase_admin import credentials, db
from PIL import Image
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference
from ultralytics import YOLO

load_dotenv()
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)
app = FastAPI(title="Body Tracker API", description="API for tracking workout")
model = YOLO("yolo11s-pose.pt", task="pose")

# cred = credentials.Certificate("serviceAccountKey.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://<YOUR_PROJECT_ID>.firebaseio.com/'  # replace with your Firebase DB URL
# })

# def insert_data(path, data):
#     """
#     Insert data into Firebase Realtime Database at the specified path.
#     :param path: str, the database path (e.g., 'users/user1')
#     :param data: dict, the data to insert
#     """
#     ref = db.reference(path)
#     ref.set(data)


# Constants for keypoint indices
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_ELBOW_IDX = 7
RIGHT_ELBOW_IDX = 8
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def save_data(user_id, workout_type, rep_count, timestamp):
    print(
        f"Saving data for user {user_id} in workout {workout_type} with {rep_count} reps at {timestamp}"
    )
    # ref = db.reference(f"/users/{user_id}/workouts/{workout_type}/{timestamp}")
    # ref.set({"reps": rep_count, "timestamp": timestamp})


def calculate_angle_numpy(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculates the angle (in degrees) at point p2 formed by the vectors
    p2->p1 and p2->p3 using NumPy.

    Args:
      p1: NumPy array (x, y) or (x, y, z) coordinates of the first point.
      p2: NumPy array (x, y) or (x, y, z) coordinates of the vertex (middle point).
      p3: NumPy array (x, y) or (x, y, z) coordinates of the third point.

    Returns:
      The angle in degrees (float). Returns 0.0 if either vector has zero length.
    """
    # Ensure inputs are NumPy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate vectors
    vec1 = p1 - p2  # Vector from p2 to p1
    vec2 = p3 - p2  # Vector from p2 to p3

    # Calculate vector norms (lengths)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle potential zero-length vectors (if points overlap)
    if norm1 == 0 or norm2 == 0:
        # print("Warning: Zero-length vector encountered. Returning 0 degrees.")
        return 0.0

    # Calculate unit vectors
    unit_vec1 = vec1 / norm1
    unit_vec2 = vec2 / norm2

    # Calculate the dot product
    dot_product = np.dot(unit_vec1, unit_vec2)

    # Clip the dot product to the valid range [-1.0, 1.0]
    # This prevents potential floating-point errors causing arccos domain issues
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians using arccos
    angle_rad = np.arccos(dot_product)

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def is_squat(keypoints: np.ndarray, confidence: np.ndarray) -> bool:
    # threshold = orig_shape[1] / 100 * 7

    # Get the coordinates of the knees and hips
    if (
        (confidence[LEFT_KNEE_IDX] < 0.5 and confidence[RIGHT_KNEE_IDX] < 0.5)
        or (confidence[LEFT_HIP_IDX] < 0.5 and confidence[RIGHT_HIP_IDX] < 0.5)
        or (confidence[LEFT_ANKLE_IDX] < 0.5 and confidence[RIGHT_ANKLE_IDX] < 0.5)
        or (
            confidence[LEFT_SHOULDER_IDX] < 0.5 and confidence[RIGHT_SHOULDER_IDX] < 0.5
        )
    ):
        return False

    left_knee = keypoints[LEFT_KNEE_IDX]
    right_knee = keypoints[RIGHT_KNEE_IDX]
    left_hip = keypoints[LEFT_HIP_IDX]
    right_hip = keypoints[RIGHT_HIP_IDX]
    left_ankle = keypoints[LEFT_ANKLE_IDX]
    right_ankle = keypoints[RIGHT_ANKLE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]

    knee = (
        (left_knee + right_knee) / 2
        if (confidence[LEFT_KNEE_IDX] > 0.5 and confidence[RIGHT_KNEE_IDX] > 0.5)
        else left_knee
        if confidence[LEFT_KNEE_IDX] > 0.5
        else right_knee
    )
    ankle = (
        (left_ankle + right_ankle) / 2
        if (confidence[LEFT_ANKLE_IDX] > 0.5 and confidence[RIGHT_ANKLE_IDX] > 0.5)
        else left_ankle
        if confidence[LEFT_ANKLE_IDX] > 0.5
        else right_ankle
    )
    hip = (
        (left_hip + right_hip) / 2
        if (confidence[LEFT_HIP_IDX] > 0.5 and confidence[RIGHT_HIP_IDX] > 0.5)
        else left_hip
        if confidence[LEFT_HIP_IDX] > 0.5
        else right_hip
    )
    shoulder = (
        (left_shoulder + right_shoulder) / 2
        if (
            confidence[LEFT_SHOULDER_IDX] > 0.5 and confidence[RIGHT_SHOULDER_IDX] > 0.5
        )
        else left_shoulder
        if confidence[LEFT_SHOULDER_IDX] > 0.5
        else right_shoulder
    )

    knee_angle = calculate_angle_numpy(ankle, knee, hip)
    hip_angle = calculate_angle_numpy(hip, knee, shoulder)

    if knee_angle < 90 and hip_angle < 90:
        return True

    # return (
    #     abs(left_hip[1] - left_knee[1]) < threshold
    #     and abs(right_hip[1] - right_knee[1]) < threshold
    # )


def is_plank(keypoints: np.ndarray, confidence: np.ndarray) -> bool:
    # threshold = orig_shape[1] / 100 * 10

    left_knee = keypoints[LEFT_KNEE_IDX]
    right_knee = keypoints[RIGHT_KNEE_IDX]
    left_hip = keypoints[LEFT_HIP_IDX]
    right_hip = keypoints[RIGHT_HIP_IDX]
    left_ankle = keypoints[LEFT_ANKLE_IDX]
    right_ankle = keypoints[RIGHT_ANKLE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]
    # Get the coordinates of the knees and hips

    if (
        (confidence[LEFT_KNEE_IDX] < 0.5 and confidence[RIGHT_KNEE_IDX] < 0.5)
        or (confidence[LEFT_HIP_IDX] < 0.5 and confidence[RIGHT_HIP_IDX] < 0.5)
        or (
            confidence[LEFT_SHOULDER_IDX] < 0.5 and confidence[RIGHT_SHOULDER_IDX] < 0.5
        )
        or (confidence[LEFT_ANKLE_IDX] < 0.5 and confidence[RIGHT_ANKLE_IDX] < 0.5)
    ):
        return False

    ankle = (
        (left_ankle + right_ankle) / 2
        if (confidence[LEFT_ANKLE_IDX] > 0.5 and confidence[RIGHT_ANKLE_IDX] > 0.5)
        else left_ankle
        if confidence[LEFT_ANKLE_IDX] > 0.5
        else right_ankle
    )

    # Calculate Knee point
    knee = (
        (left_knee + right_knee) / 2
        if (confidence[LEFT_KNEE_IDX] > 0.5 and confidence[RIGHT_KNEE_IDX] > 0.5)
        else left_knee
        if confidence[LEFT_KNEE_IDX] > 0.5
        else right_knee
    )

    # Calculate hip point
    hip = (
        (left_hip + right_hip) / 2
        if (confidence[LEFT_HIP_IDX] > 0.5 and confidence[RIGHT_HIP_IDX] > 0.5)
        else left_hip
        if confidence[LEFT_HIP_IDX] > 0.5
        else right_hip
    )

    shoulder = (
        (left_shoulder + right_shoulder) / 2
        if (
            confidence[LEFT_SHOULDER_IDX] > 0.5 and confidence[RIGHT_SHOULDER_IDX] > 0.5
        )
        else left_shoulder
        if confidence[LEFT_SHOULDER_IDX] > 0.5
        else right_shoulder
    )

    # Calculate Knee Angles
    knee_angle = calculate_angle_numpy(ankle, knee, hip)

    # Calculate Hip Angles (Torso-Thigh)
    hip_angle = calculate_angle_numpy(shoulder, hip, knee)

    # print(knee_angle, hip_angle)
    return knee_angle >= 155 and hip_angle >= 155


def is_pushup(keypoints: np.ndarray, confidence: np.ndarray) -> bool:
    left_knee = keypoints[LEFT_KNEE_IDX]
    right_knee = keypoints[RIGHT_KNEE_IDX]
    left_hip = keypoints[LEFT_HIP_IDX]
    right_hip = keypoints[RIGHT_HIP_IDX]
    left_ankle = keypoints[LEFT_ANKLE_IDX]
    right_ankle = keypoints[RIGHT_ANKLE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]
    left_elbow = keypoints[LEFT_ELBOW_IDX]
    right_elbow = keypoints[RIGHT_ELBOW_IDX]
    left_wrist = keypoints[LEFT_WRIST_IDX]
    right_wrist = keypoints[RIGHT_WRIST_IDX]

    if (
        (confidence[LEFT_KNEE_IDX] < 0.5 and confidence[RIGHT_KNEE_IDX] < 0.5)
        or (confidence[LEFT_HIP_IDX] < 0.5 and confidence[RIGHT_HIP_IDX] < 0.5)
        or (
            confidence[LEFT_SHOULDER_IDX] < 0.5 and confidence[RIGHT_SHOULDER_IDX] < 0.5
        )
        or (confidence[LEFT_ELBOW_IDX] < 0.5 and confidence[RIGHT_ELBOW_IDX] < 0.5)
        or (confidence[LEFT_WRIST_IDX] < 0.5 and confidence[RIGHT_WRIST_IDX] < 0.5)
    ):
        return False

    knee = (
        (left_knee + right_knee) / 2
        if (confidence[LEFT_KNEE_IDX] > 0.5 and confidence[RIGHT_KNEE_IDX] > 0.5)
        else left_knee
        if confidence[LEFT_KNEE_IDX] > 0.5
        else right_knee
    )
    ankle = (
        (left_ankle + right_ankle) / 2
        if (confidence[LEFT_ANKLE_IDX] > 0.5 and confidence[RIGHT_ANKLE_IDX] > 0.5)
        else left_ankle
        if confidence[LEFT_ANKLE_IDX] > 0.5
        else right_ankle
    )
    hip = (
        (left_hip + right_hip) / 2
        if (confidence[LEFT_HIP_IDX] > 0.5 and confidence[RIGHT_HIP_IDX] > 0.5)
        else left_hip
        if confidence[LEFT_HIP_IDX] > 0.5
        else right_hip
    )
    shoulder = (
        (left_shoulder + right_shoulder) / 2
        if (
            confidence[LEFT_SHOULDER_IDX] > 0.5 and confidence[RIGHT_SHOULDER_IDX] > 0.5
        )
        else left_shoulder
        if confidence[LEFT_SHOULDER_IDX] > 0.5
        else right_shoulder
    )
    elbow = (
        (left_elbow + right_elbow) / 2
        if (confidence[LEFT_ELBOW_IDX] > 0.5 and confidence[RIGHT_ELBOW_IDX] > 0.5)
        else left_elbow
        if confidence[LEFT_ELBOW_IDX] > 0.5
        else right_elbow
    )
    wrist = (
        (left_wrist + right_wrist) / 2
        if (confidence[LEFT_WRIST_IDX] > 0.5 and confidence[RIGHT_WRIST_IDX] > 0.5)
        else left_wrist
        if confidence[LEFT_WRIST_IDX] > 0.5
        else right_wrist
    )

    knee_angle = calculate_angle_numpy(ankle, knee, hip)
    hip_angle = calculate_angle_numpy(hip, knee, shoulder)
    elbow_angle = calculate_angle_numpy(shoulder, elbow, wrist)

    return (
        hip[1] > knee[1]
        and knee[1] > ankle[1]
        and knee_angle >= 170
        and hip_angle >= 170
        and elbow_angle >= 170
    )


def is_lunges(keypoints: np.ndarray, confidence: np.ndarray) -> bool:
    left_knee = keypoints[LEFT_KNEE_IDX]
    right_knee = keypoints[RIGHT_KNEE_IDX]
    left_hip = keypoints[LEFT_HIP_IDX]
    right_hip = keypoints[RIGHT_HIP_IDX]
    left_ankle = keypoints[LEFT_ANKLE_IDX]
    right_ankle = keypoints[RIGHT_ANKLE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]
    left_ear = keypoints[LEFT_EAR_IDX]
    right_ear = keypoints[RIGHT_EAR_IDX]

    if (
        confidence[LEFT_KNEE_IDX] < 0.5
        or confidence[RIGHT_KNEE_IDX] < 0.5
        or (confidence[LEFT_HIP_IDX] < 0.5 and confidence[RIGHT_HIP_IDX] < 0.5)
        or (confidence[LEFT_ANKLE_IDX] < 0.5 and confidence[RIGHT_ANKLE_IDX] < 0.5)
        or (
            confidence[LEFT_SHOULDER_IDX] < 0.5 and confidence[RIGHT_SHOULDER_IDX] < 0.5
        )
        or (confidence[LEFT_EAR_IDX] < 0.5 and confidence[RIGHT_EAR_IDX] < 0.5)
    ):
        return False

    hip = (
        (left_hip + right_hip) / 2
        if (confidence[LEFT_HIP_IDX] > 0.5 and confidence[RIGHT_HIP_IDX] > 0.5)
        else left_hip
        if confidence[LEFT_HIP_IDX] > 0.5
        else right_hip
    )
    shoulder = (
        (left_shoulder + right_shoulder) / 2
        if (
            confidence[LEFT_SHOULDER_IDX] > 0.5 and confidence[RIGHT_SHOULDER_IDX] > 0.5
        )
        else left_shoulder
        if confidence[LEFT_SHOULDER_IDX] > 0.5
        else right_shoulder
    )
    ear = (
        (left_ear + right_ear) / 2
        if (confidence[LEFT_EAR_IDX] > 0.5 and confidence[RIGHT_EAR_IDX] > 0.5)
        else left_ear
        if confidence[LEFT_EAR_IDX] > 0.5
        else right_ear
    )

    left_knee_angle = calculate_angle_numpy(left_ankle, left_knee, hip)
    right_knee_angle = calculate_angle_numpy(right_ankle, right_knee, hip)
    left_knee_hip_angle = calculate_angle_numpy(hip, left_knee, shoulder)
    right_knee_hip_angle = calculate_angle_numpy(hip, right_knee, shoulder)
    shoulder_ear_angle = calculate_angle_numpy(shoulder, ear, hip)

    return (
        shoulder_ear_angle >= 170
        and (left_knee_angle < 80 and left_knee_angle > 120)
        and (right_knee_angle < 80 and right_knee_angle > 120)
        and (
            (left_knee_hip_angle < 80 and left_knee_hip_angle > 100)
            or (right_knee_hip_angle < 80 and right_knee_hip_angle > 100)
        )
    )


def is_jumping_jacks(keypoints: np.ndarray, confidence: np.ndarray) -> bool:
    left_knee = keypoints[LEFT_KNEE_IDX]
    right_knee = keypoints[RIGHT_KNEE_IDX]
    left_ankle = keypoints[LEFT_ANKLE_IDX]
    right_ankle = keypoints[RIGHT_ANKLE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]
    left_wrist = keypoints[LEFT_WRIST_IDX]
    right_wrist = keypoints[RIGHT_WRIST_IDX]
    left_elbow = keypoints[LEFT_ELBOW_IDX]
    right_elbow = keypoints[RIGHT_ELBOW_IDX]
    left_hip = keypoints[LEFT_HIP_IDX]
    right_hip = keypoints[RIGHT_HIP_IDX]

    if (
        confidence[LEFT_KNEE_IDX] < 0.5
        or confidence[RIGHT_KNEE_IDX] < 0.5
        or confidence[LEFT_ANKLE_IDX] < 0.5
        or confidence[RIGHT_ANKLE_IDX] < 0.5
        or confidence[LEFT_SHOULDER_IDX] < 0.5
        or confidence[RIGHT_SHOULDER_IDX] < 0.5
        or confidence[LEFT_ELBOW_IDX] < 0.5
        or confidence[RIGHT_ELBOW_IDX] < 0.5
        or confidence[LEFT_WRIST_IDX] < 0.5
        or confidence[RIGHT_WRIST_IDX] < 0.5
        or confidence[LEFT_HIP_IDX] < 0.5
        or confidence[RIGHT_HIP_IDX] < 0.5
    ):
        return False

    hip = (left_hip + right_hip) / 2

    left_elbow_angle = calculate_angle_numpy(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle_numpy(right_shoulder, right_elbow, right_wrist)
    left_knee_angle = calculate_angle_numpy(left_ankle, left_knee, hip)
    right_knee_angle = calculate_angle_numpy(right_ankle, right_knee, hip)
    hip_angle = calculate_angle_numpy(hip, left_knee, right_knee)

    return (
        hip_angle > 60
        and left_knee_angle >= 170
        and right_knee_angle >= 170
        and left_elbow_angle >= 120
        and right_elbow_angle >= 120
    )


class ConnectionInfo(BaseModel):
    frame_count: int = 0
    squat_count: int = 0
    plank_count: int = 0
    pushup_count: int = 0
    lunges_count: int = 0
    jumping_jacks_count: int = 0
    timestamp: float = 0


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: dict[WebSocket, ConnectionInfo] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_info[websocket] = ConnectionInfo()
        self.connection_info[websocket].timestamp = time.time()

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
                print("time elapsed:", (time.time() - manager.connection_info[websocket].timestamp) * 1000, "ms")
                manager.connection_info[websocket].timestamp = time.time()

                # print("Frame count:", manager.connection_info[websocket].frame_count)
                image = Image.open(io.BytesIO(frame))
                results = model(image, verbose=False, device=device)


                squat_bool = False
                lunges_bool = False
                plank_bool = False
                pushup_bool = False
                jumping_jacks_bool = False

                if (
                    len(results) > 0
                    and results[0] is not None
                    and results[0].names[0] == "person"
                    and results[0].keypoints is not None
                    and results[0].keypoints.conf is not None
                    and results[0].keypoints.orig_shape is not None
                ):
                    if torch.cuda.is_available():
                        keypoints = results[0].keypoints.xy[0].cpu().numpy()
                        confidence = results[0].keypoints.conf[0].cpu().numpy()
                    else:
                        keypoints = np.array(results[0].keypoints.xy[0])
                        confidence = np.array(results[0].keypoints.conf[0])
                    # orig_shape = np.array(results[0].keypoints.orig_shape)

                    squat_bool = bool(is_squat(keypoints, confidence))
                    lunges_bool = bool(is_lunges(keypoints, confidence))
                    plank_bool = bool(is_plank(keypoints, confidence))
                    pushup_bool = bool(is_pushup(keypoints, confidence))
                    jumping_jacks_bool = bool(is_jumping_jacks(keypoints, confidence))
                    if squat_bool:
                        manager.connection_info[websocket].squat_count += 1
                    if lunges_bool:
                        manager.connection_info[websocket].lunges_count += 1
                    if plank_bool:
                        manager.connection_info[websocket].plank_count += 1
                    if pushup_bool:
                        manager.connection_info[websocket].pushup_count += 1
                    if jumping_jacks_bool:
                        manager.connection_info[websocket].jumping_jacks_count += 1

                keypoints_data = {
                    "keypoints": keypoints.tolist(),
                    "confidence": confidence.tolist(),
                    # "orig_shape": orig_shape.tolist(),
                    "frame_count": manager.connection_info[websocket].frame_count,
                    "squat_count": manager.connection_info[websocket].squat_count,
                    "plank_count": manager.connection_info[websocket].plank_count,
                    "pushup_count": manager.connection_info[websocket].pushup_count,
                    "lunges_count": manager.connection_info[websocket].lunges_count,
                    "jumping_jacks_count": manager.connection_info[
                        websocket
                    ].jumping_jacks_count,
                    "squat_bool": squat_bool,
                    "lunges_bool": lunges_bool,
                    "plank_bool": plank_bool,
                    "pushup_bool": pushup_bool,
                    "jumping_jacks_bool": jumping_jacks_bool,
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
