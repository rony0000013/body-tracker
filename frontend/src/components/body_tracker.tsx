import { createEffect, createSignal, onCleanup, onMount, Show } from "solid-js";
export const BODY_PARTS = [
    // Face
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    // Torso
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],
    // Left Arm
    [5, 7],
    [7, 9],
    // Right Arm
    [6, 8],
    [8, 10],
    // Left Leg
    [11, 13],
    [13, 15],
    // Right Leg
    [12, 14],
    [14, 16],
];

const WEBSOCKET_URL = import.meta.env.PUBLIC_WEBSOCKET_URL;
// export const WEBSOCKET_URL = "ws://localhost:8000/ws"; // Replace with your actual WebSocket URL
export const TARGET_FPS = 12;
export const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

// Define the structure of the data received from the WebSocket
interface PoseData {
    keypoints: [number, number][]; // Array of [x, y] tuples
    confidence: number[];
    frame_count: number;
    squat_count: number;
    plank_count: number;
    pushup_count: number;
    lunges_count: number;
    jumping_jacks_count: number;
    squat_bool: boolean;
    lunges_bool: boolean;
    plank_bool: boolean;
    pushup_bool: boolean;
    jumping_jacks_bool: boolean;
}

function App(props: { workout: string; userId: string }) {

    const workout = props.workout;
    const userId = props.userId;
    let videoRef: HTMLVideoElement | undefined;
    let canvasRef: HTMLCanvasElement | undefined;
    let captureCanvas: HTMLCanvasElement | undefined; // Offscreen canvas for frame capture
    let intervalId: NodeJS.Timeout | undefined; // To store the interval timer ID
    let stream: MediaStream | null = null;

    const [isConnected, setIsConnected] = createSignal(false);
    const [ws, setWs] = createSignal<WebSocket | null>(null);
    const [poseData, setPoseData] = createSignal<PoseData | null>(null);
    const [error, setError] = createSignal<string | null>(null);
    const [isStreaming, setIsStreaming] = createSignal(false);
    const [isVideoReady, setIsVideoReady] = createSignal(false);

    // --- WebSocket Handling ---

    const connectWebSocket = () => {
        setError(null);
        console.log("Attempting to connect to WebSocket...");
        const socket = new WebSocket(WEBSOCKET_URL);

        socket.onopen = () => {
            console.log("WebSocket Connected");
            setIsConnected(true);
            setWs(socket);
            setError(null);
            // Automatically start streaming if video is ready
            if (isVideoReady() && videoRef) {
                startSendingFrames();
            }
        };

        socket.onmessage = (event) => {
            try {
                const data: PoseData = JSON.parse(event.data as string);
                // Basic validation (can be more robust)
                if (data.keypoints && Array.isArray(data.keypoints)) {
                    setPoseData(data);
                    setError(null); // Clear previous errors on successful message
                } else {
                    console.warn("Received invalid data structure:", data);
                    // setError("Received invalid data structure from server.");
                    // Don't set error here usually, just ignore the malformed message
                }
            } catch (e) {
                console.error("Failed to parse WebSocket message:", e);
                setError("Error parsing data from server.");
                setPoseData(null); // Clear potentially stale data
            }
        };

        socket.onerror = (event) => {
            console.error("WebSocket Error:", event);
            setError(
                `WebSocket error. Check console and if the server at ${WEBSOCKET_URL} is running.`,
            );
            setIsConnected(false);
            setWs(null);
            stopSendingFrames(); // Stop trying to send if connection failed
        };

        socket.onclose = (event) => {
            console.log("WebSocket Disconnected:", event.reason);
            setIsConnected(false);
            setWs(null);
            setPoseData(null); // Clear data on disconnect
            if (!event.wasClean) {
                setError("WebSocket connection closed unexpectedly.");
            }
            stopSendingFrames();
            // Optional: Implement automatic reconnection logic here
        };
    };

    // --- Camera Handling ---

    const startCamera = async () => {
        setError(null);
        if (stream) { // If already streaming, stop the old one first
            stream.getTracks().forEach((track) => track.stop());
        }
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    // Optional: request specific dimensions
                    // width: { ideal: 640 },
                    // height: { ideal: 480 }
                },
                audio: false,
            });
            if (videoRef) {
                videoRef.srcObject = stream;
                videoRef.onloadedmetadata = () => {
                    console.log("Video metadata loaded");
                    videoRef?.play().catch((e) =>
                        console.error("Video play failed:", e)
                    );
                    setIsVideoReady(true);
                    // Connect WS or start streaming if WS already connected
                    if (ws() && ws()?.readyState === WebSocket.OPEN) {
                        startSendingFrames();
                    } else if (!ws()) {
                        connectWebSocket(); // Attempt connection now that video is ready
                    }
                };
                videoRef.oncanplay = () => {
                    console.log("Video can play");
                    // Sometimes loadedmetadata isn't enough, ensure it can play
                    if (!isVideoReady()) setIsVideoReady(true);
                };
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
            setError(
                "Could not access camera. Please ensure permission is granted and no other app is using it.",
            );
            setIsVideoReady(false);
        }
    };

    const stopCamera = () => {
        stopSendingFrames(); // Stop sending frames first
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }
        if (videoRef) {
            videoRef.srcObject = null;
        }
        setIsVideoReady(false);
        setPoseData(null); // Clear pose data when camera stops
        clearCanvas(); // Clear the overlay
    };

    const startCameraAndConnect = () => {
        // Start camera/stream logic
        // Connect websocket logic
        if (!isStreaming()) startCamera(); // Or your function to start camera
        if (!isConnected()) connectWebSocket(); // Or your function to connect websocket
    };

    const stopCameraAndDisconnect = () => {
        // Stop camera/stream logic
        // Disconnect websocket logic
        if (isStreaming()) stopCamera(); // Or your function to stop camera
        if (isConnected()) disconnectWebSocket(); // Or your function to disconnect websocket
    };

    const disconnectWebSocket = () => {
        const socket = ws();
        if (socket && socket.readyState !== WebSocket.CLOSED) {
            socket.close();
        }
        setIsConnected(false);
        setWs(null);
        setPoseData(null); // Clear pose data on disconnect
        setError(null);
        stopSendingFrames();
        // Optionally: clearCanvas(); if you want to clear the overlay
    };

    // --- Frame Sending Logic ---

    const sendFrame = () => {
        if (
            !videoRef || !captureCanvas || !ws() ||
            ws()?.readyState !== WebSocket.OPEN || !isVideoReady()
        ) {
            // console.warn("Skipping frame send: Conditions not met.");
            return;
        }

        // Ensure capture canvas dimensions match video's intrinsic size
        const videoWidth = videoRef.videoWidth;
        const videoHeight = videoRef.videoHeight;

        if (videoWidth === 0 || videoHeight === 0) {
            // console.warn("Skipping frame send: Video dimensions are zero.");
            return; // Video not ready yet
        }

        if (
            captureCanvas.width !== videoWidth ||
            captureCanvas.height !== videoHeight
        ) {
            captureCanvas.width = videoWidth;
            captureCanvas.height = videoHeight;
            console.log(
                `Capture canvas resized to: ${videoWidth}x${videoHeight}`,
            );
        }

        const context = captureCanvas.getContext("2d");
        if (!context) return;

        // Draw the current video frame onto the hidden canvas
        context.drawImage(videoRef, 0, 0, videoWidth, videoHeight);

        // Send the raw binary image data via WebSocket
        if (ws()?.readyState === WebSocket.OPEN) {
            captureCanvas.toBlob((blob) => {
                if (blob) {
                    blob.arrayBuffer().then((buffer) => ws()?.send(buffer));
                }
            }, "image/png");
        } else {
            console.warn("WebSocket not open, cannot send frame.");
            // Consider stopping the interval if WS is closed unexpectedly
            // stopSendingFrames();
            // setError("WebSocket closed. Cannot send data.");
        }
    };

    const startSendingFrames = () => {
        if (intervalId) clearInterval(intervalId); // Clear existing interval
        if (!isVideoReady()) {
            console.warn("Cannot start sending frames: Video not ready.");
            return;
        }
        if (!ws() || ws()?.readyState !== WebSocket.OPEN) {
            console.warn(
                "Cannot start sending frames: WebSocket not connected.",
            );
            return;
        }

        console.log(
            `Starting frame sending interval (${
                (1000 / FRAME_INTERVAL_MS).toFixed(1)
            } FPS)`,
        );
        // Create the hidden canvas for capturing frames if it doesn't exist
        if (!captureCanvas) {
            captureCanvas = document.createElement("canvas");
        }

        intervalId = setInterval(sendFrame, FRAME_INTERVAL_MS);
        setIsStreaming(true);
    };

    const stopSendingFrames = () => {
        if (intervalId) {
            console.log("Stopping frame sending interval.");
            clearInterval(intervalId);
            intervalId = undefined;
        }
        setIsStreaming(false);
    };

    // --- Drawing Logic ---

    const clearCanvas = () => {
        if (canvasRef) {
            const ctx = canvasRef.getContext("2d");
            if (ctx) {
                // Also ensure internal size is set correctly before clearing if needed
                // canvasRef.width = videoRef?.videoWidth || canvasRef.width;
                // canvasRef.height = videoRef?.videoHeight || canvasRef.height;
                ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
            }
        }
    };

    const drawOverlay = (data: PoseData) => {
        if (!canvasRef || !videoRef || !isVideoReady()) return;

        const ctx = canvasRef.getContext("2d");
        if (!ctx) return;

        const videoWidth = videoRef.videoWidth;
        const videoHeight = videoRef.videoHeight;

        // Match canvas internal resolution to video resolution
        // This is crucial for correct coordinate mapping
        if (
            canvasRef.width !== videoWidth || canvasRef.height !== videoHeight
        ) {
            canvasRef.width = videoWidth;
            canvasRef.height = videoHeight;
            console.log(
                `Overlay canvas resized to: ${videoWidth}x${videoHeight}`,
            );
        }

        // Clear previous drawings
        ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);

        ctx.drawImage(videoRef, 0, 0, videoWidth, videoHeight);

        if (!data || !data.keypoints) return; // No data to draw

        const keypoints = data.keypoints as [number, number][];
        const confidence = data.confidence as number[]; // Optional: Use confidence for visibility/color

        // --- Draw Lines (Skeleton) ---
        ctx.strokeStyle = "lime"; // Color for lines
        ctx.lineWidth = 3;

        BODY_PARTS.forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            const conf1 = confidence[i] ?? 0; // Default to 1.0 if no confidence
            const conf2 = confidence[j] ?? 0;

            // Only draw if both keypoints are reasonably confident (e.g., > 0.2)
            // and coordinates are valid numbers
            if (
                kp1[0] !== 0 && kp1[1] !== 0 && kp2[0] !== 0 && kp2[1] !== 0 &&
                conf1 > 0.2 && conf2 > 0.2
            ) {
                ctx.beginPath();
                ctx.moveTo(kp1[0], kp1[1]);
                ctx.lineTo(kp2[0], kp2[1]);
                ctx.stroke();
            }
        });

        // --- Draw Points (Keypoints) ---
        ctx.fillStyle = "red"; // Color for points
        const pointRadius = 5;

        keypoints.forEach((kp, index) => {
            const conf = confidence[index] ?? 0;
            if (kp && kp[0] !== 0 && kp[1] !== 0 && conf > 0.2) { // Check for valid coordinates and confidence
                ctx.beginPath();
                ctx.arc(kp[0], kp[1], pointRadius, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
            }
        });
    };

    // --- Effects and Lifecycle ---

    // Effect to draw overlay whenever poseData changes
    createEffect(() => {
        const currentPoseData = poseData();
        if (currentPoseData && isVideoReady()) {
            drawOverlay(currentPoseData);
        } else {
            // Clear canvas if no pose data or video isn't ready
            clearCanvas();
        }
    });

    // Clean up on component unmount
    onCleanup(() => {
        console.log("Cleaning up App component...");
        stopCamera(); // Stops stream and sending frames
        ws()?.close(); // Close WebSocket connection
        setWs(null);
        if (intervalId) clearInterval(intervalId);
    });

    // Optionally, attempt to start camera on mount (requires user interaction usually)
    // onMount(() => {
    //     startCamera(); // Might be blocked by browser autoplay policies
    // });

    return (
        <div class="container flex flex-col items-center h-full">
            <h1 class="text-primary text-2xl">
                Workout Counter
            </h1>

            <div class="video-container w-[300px] h-[300px] mb-6 rounded-lg border-primary ">
                <video
                    ref={videoRef}
                    class="video-feed hidden "
                    playsinline={true}
                    muted={true}
                    autoplay={true}
                />
                <canvas
                    ref={canvasRef}
                    class="overlay-canvas  w-[300px] h-[300px] rounded-lg border-primary  pointer-events-none"
                />
            </div>

            <div class="controls flex justify-center gap-4 mb-6">
                <button
                    class="btn btn-success"
                    onClick={startCameraAndConnect}
                    disabled={isStreaming() || isConnected()}
                >
                    Start Camera
                </button>
                <button
                    class="btn btn-error"
                    onClick={stopCameraAndDisconnect}
                    disabled={!isStreaming() && !isConnected()}
                >
                    Stop Camera
                </button>
            </div>

            <div class="card flex flex-col items-center mb-6">
                <h4 class="text-base-content">
                    Workout Type:
                    <span class="badge badge-info ml-2">{props.workout}</span>
                </h4>
                <h4 class="text-base-content">
                    User ID:
                    <span class="badge badge-secondary ml-2">
                        {props.userId}
                    </span>
                </h4>
                <Show when={poseData() && isConnected()}>
                    <div class="flex gap-2">
                        <Show when={workout === "squat"}>
                            Squats:<span class="countdown">
                                {poseData()?.squat_count}
                            </span>
                        </Show>
                        <Show when={workout === "plank"}>
                            Plank:<span class="countdown">
                                {(poseData()?.plank_count as number) / 24}
                            </span>
                        </Show>
                        <Show when={workout === "pushup"}>
                            Pushups:<span class="countdown">
                                {poseData()?.pushup_count}
                            </span>
                        </Show>
                        <Show when={workout === "lunges"}>
                            Lunges:<span class="countdown">
                                {poseData()?.lunges_count}
                            </span>
                        </Show>
                        <Show when={workout === "jumping-jack"}>
                            Jumping Jacks:<span class="countdown">
                                {poseData()?.jumping_jacks_count}
                            </span>
                        </Show>
                    </div>
                </Show>
                <div class="mt-2 space-y-1">
                    <div>
                        <span class="text-base-content">
                            WebSocket Status:
                        </span>
                        <Show
                            when={isConnected()}
                            fallback={
                                <span class="badge badge-error ml-2">
                                    Disconnected
                                </span>
                            }
                        >
                            <span class="badge badge-success ml-2">
                                Connected
                            </span>
                        </Show>
                    </div>
                    <div>
                        <span class="text-base-content">
                            Camera Status:
                        </span>
                        <Show
                            when={isVideoReady()}
                            fallback={
                                <span class="badge badge-error ml-2">
                                    Stopped / Not Ready
                                </span>
                            }
                        >
                            <span class="badge badge-success ml-2">
                                Ready
                            </span>
                        </Show>
                    </div>
                    <div>
                        <span class="text-base-content">
                            Streaming Status:
                        </span>
                        <Show
                            when={isStreaming()}
                            fallback={
                                <span class="badge badge-error ml-2">
                                    Stopped
                                </span>
                            }
                        >
                            <span class="badge badge-success ml-2">
                                Sending Frames
                            </span>
                        </Show>
                    </div>
                </div>
            </div>

            <Show when={error()}>
                <p class="alert alert-error mb-4">Error: {error()}</p>
            </Show>
        </div>
    );
}

export default App;
