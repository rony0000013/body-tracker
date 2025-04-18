import { createSignal, onCleanup, onMount } from "solid-js";
import type { Accessor, Component, Setter } from "solid-js"; // Import specific types if needed

// Define the props interface
interface BodyTrackerProps {
    workout: string;
    userId: string;
}

// Define the structure of the data received from the WebSocket
interface ParsedData {
    keypoints: [number, number][];
    confidence: number[];
    orig_shape: [number, number]; // Assuming this might be needed later
    squat_count: number;
}

const BodyTracker: Component<BodyTrackerProps> = (props) => {
    // --- Refs for DOM Elements ---
    let videoRef: HTMLVideoElement | undefined;
    let canvasRef: HTMLCanvasElement | undefined;
    let connectBtnRef: HTMLButtonElement | undefined;
    let disconnectBtnRef: HTMLButtonElement | undefined;
    let fpsRef: HTMLDivElement | undefined;
    let squatCountRef: HTMLDivElement | undefined;

    // --- Reactive State ---
    const [isConnected, setIsConnected] = createSignal(false);
    const [fps, setFps] = createSignal(0);
    const [squatCount, setSquatCount] = createSignal(0);

    // --- Non-Reactive State & Constants ---
    let ctx: CanvasRenderingContext2D | null = null;
    let ws: WebSocket | null = null;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    let frameCount = 0;
    let startTime = 0;
    const FRAME_INTERVAL = 1000 / 12; // Target ~12 FPS for sending
    let isManuallyDisconnected = false;
    let animationFrameId: number | null = null; // To manage the processing loop

    // --- WebSocket Connection Logic ---
    const connectWebSocket = () => {
        if (isConnected()) {
            console.log("WebSocket is already connected");
            return;
        }

        // Close existing connection if any before reconnecting
        if (ws) {
            console.log(
                "Closing previous WebSocket connection before reconnecting.",
            );
            ws.close();
            ws = null;
        }

        // Reset manual disconnect flag when attempting connection
        isManuallyDisconnected = false;
        // Use a secure WebSocket (wss://) if your server supports it, otherwise ws://
        // Make sure the IP and port are correct.
        ws = new WebSocket(`${process.env.WEBSOCKET_URL}`);
        console.log(`Connecting to WebSocket... ${process.env.WEBSOCKET_URL}`);

        ws.onopen = () => {
            console.log("Connected to WebSocket server");
            setIsConnected(true);
            retryCount = 0; // Reset retries on successful connection
            if (connectBtnRef) connectBtnRef.disabled = true;
            if (disconnectBtnRef) disconnectBtnRef.disabled = false;
            startTime = Date.now(); // Reset start time for FPS calculation
            frameCount = 0; // Reset frame count
            requestProcessFrame(); // Start processing frames
        };

        ws.onmessage = (event: MessageEvent) => {
            try {
                if (!canvasRef || !ctx || !videoRef) return;

                const parsedData: ParsedData = JSON.parse(event.data);
                const { keypoints, confidence, squat_count } = parsedData;

                // Update squat count state
                setSquatCount(squat_count);

                // --- Drawing Logic ---
                // Ensure canvas dimensions match video dimensions for accurate drawing
                if (
                    canvasRef.width !== videoRef.videoWidth ||
                    canvasRef.height !== videoRef.videoHeight
                ) {
                    canvasRef.width = videoRef.videoWidth;
                    canvasRef.height = videoRef.videoHeight;
                }

                // Draw the current video frame onto the canvas
                ctx.drawImage(
                    videoRef,
                    0,
                    0,
                    canvasRef.width,
                    canvasRef.height,
                );

                // Set drawing styles
                ctx.strokeStyle = "red";
                ctx.fillStyle = "red";
                ctx.lineWidth = 2;

                // Draw keypoints
                keypoints.forEach((point, index) => {
                    if (confidence[index] > 0.5) { // Confidence threshold
                        const [x, y] = point;
                        ctx!.beginPath();
                        ctx!.arc(x, y, 5, 0, Math.PI * 2); // Draw a small circle
                        ctx!.fill();
                    }
                });

                // Define skeleton connections (indices of keypoints)
                const skeleton: [number, number][] = [
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 4], // Head
                    [5, 6],
                    [5, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10], // Torso/Arms
                    [11, 12],
                    [11, 13],
                    [12, 14],
                    [13, 15],
                    [14, 16], // Legs
                    [5, 11],
                    [6, 12], // Connect shoulders to hips
                ];

                // Draw skeleton lines
                ctx.beginPath(); // Start a single path for all lines for potential efficiency
                for (const [a, b] of skeleton) {
                    if (confidence[a] > 0.5 && confidence[b] > 0.5) {
                        ctx.moveTo(keypoints[a][0], keypoints[a][1]);
                        ctx.lineTo(keypoints[b][0], keypoints[b][1]);
                    }
                }
                ctx.stroke(); // Draw all lines added to the path
            } catch (err) {
                console.error("Error processing message:", err);
                // Optionally disconnect or handle error appropriately
                // handleDisconnect(); // Decide if an error here should disconnect
            }
        };

        ws.onerror = (error: Event) => {
            console.error("WebSocket error:", error);
            handleDisconnect(false); // Assume it wasn't a manual disconnect
        };

        ws.onclose = (event: CloseEvent) => {
            console.log(
                "WebSocket connection closed:",
                event.code,
                event.reason,
                "Clean close:",
                event.wasClean,
            );
            // Only attempt reconnect if not manually disconnected and retries remain
            const shouldRetry = !isManuallyDisconnected &&
                retryCount < MAX_RETRIES;
            handleDisconnect(isManuallyDisconnected); // Pass manual state

            if (shouldRetry) {
                retryCount++;
                console.log(
                    `Attempting to reconnect (${retryCount}/${MAX_RETRIES})...`,
                );
                setTimeout(
                    connectWebSocket,
                    2000 * Math.pow(2, retryCount - 1),
                ); // Exponential backoff
            } else if (!isManuallyDisconnected) {
                console.error(
                    "WebSocket disconnected permanently after retries or due to an unclean close.",
                );
                // Optionally inform the user
            }
        };
    };

    // --- Disconnect Logic ---
    const handleDisconnect = (manual: boolean) => {
        console.log(`Disconnected from WebSocket. Manual: ${manual}`);
        isManuallyDisconnected = manual; // Set the flag based on how disconnect was initiated
        setIsConnected(false);
        if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.close();
        }
        ws = null; // Clear the WebSocket instance

        if (connectBtnRef) connectBtnRef.disabled = false;
        if (disconnectBtnRef) disconnectBtnRef.disabled = true;

        // Stop the frame processing loop
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        // Clear canvas? Optional.
        if (ctx && canvasRef) {
            ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);
        }
    };

    const manualDisconnect = () => {
        if (ws) {
            isManuallyDisconnected = true; // Set flag before closing
            ws.close();
            // handleDisconnect will be called by ws.onclose
        } else {
            handleDisconnect(true); // Ensure UI updates if ws was already null
        }
    };

    // --- Frame Processing Logic ---
    const processFrame = async () => {
        if (
            !isConnected() || !videoRef || !canvasRef || !ctx || !ws ||
            ws.readyState !== WebSocket.OPEN ||
            videoRef.readyState < videoRef.HAVE_METADATA ||
            videoRef.videoWidth === 0
        ) {
            // If not ready, try again on the next animation frame if still supposed to be connected
            if (isConnected()) {
                animationFrameId = requestAnimationFrame(processFrame);
            } else {
                animationFrameId = null; // Ensure loop stops if disconnected
            }
            return;
        }

        const now = Date.now();
        // Throttle sending frames based on FRAME_INTERVAL
        // This simple throttling might still send bursts. More complex timing could be used.
        // A simpler approach is to just send on every animation frame and let the server handle rate limiting if needed,
        // or use setTimeout like the original code if strict FPS limiting is desired.

        // Draw video to canvas (necessary for getting the blob)
        // No need to set canvas size here if it's done in onmessage or assumes video size doesn't change.
        if (
            canvasRef.width !== videoRef.videoWidth ||
            canvasRef.height !== videoRef.videoHeight
        ) {
            canvasRef.width = videoRef.videoWidth;
            canvasRef.height = videoRef.videoHeight;
        }
        ctx.drawImage(videoRef, 0, 0, canvasRef.width, canvasRef.height);

        // Send frame as Blob
        canvasRef.toBlob(
            (blob) => {
                if (
                    blob && ws && ws.readyState === WebSocket.OPEN &&
                    isConnected()
                ) {
                    ws.send(blob);
                    frameCount++;
                    const elapsed = Date.now() - startTime;
                    if (elapsed > 1000) { // Update FPS roughly every second
                        setFps(Math.round((frameCount * 1000) / elapsed));
                        // Optional: Reset for rolling average over the last second
                        // startTime = Date.now();
                        // frameCount = 0;
                    }
                } else {
                    // console.log("Skipping send: Blob creation failed or WebSocket not open/connected");
                }
                // Schedule the next frame processing
                if (isConnected()) { // Only continue if still connected
                    animationFrameId = requestAnimationFrame(processFrame);
                } else {
                    animationFrameId = null;
                }
            },
            "image/jpeg",
            0.8,
        ); // Send as JPEG with quality 0.8
    };

    // Use requestAnimationFrame for smoother loop tied to display refresh rate
    const requestProcessFrame = () => {
        if (animationFrameId === null) { // Prevent multiple loops
            animationFrameId = requestAnimationFrame(processFrame);
        }
    };

    // --- Component Lifecycle ---
    onMount(async () => {
        console.log("BodyTracker component mounted");
        console.log("Workout:", props.workout, "User ID:", props.userId);

        if (!videoRef || !canvasRef) {
            console.error("Video or Canvas ref not available on mount.");
            return;
        }

        ctx = canvasRef.getContext("2d");
        if (!ctx) {
            console.error("Failed to get 2D context from canvas.");
            return;
        }

        try {
            console.log("Requesting video stream...");
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    // Optional: Add constraints like facingMode
                    // facingMode: "user" // or "environment"
                    width: { ideal: 640 }, // Request a reasonable size
                    height: { ideal: 480 },
                },
                audio: false, // Don't request audio
            });
            videoRef.srcObject = stream;
            videoRef.play().catch((e) =>
                console.error("Video play failed:", e)
            ); // Start playing the video

            // Wait for video metadata to load to get dimensions
            videoRef.onloadedmetadata = () => {
                console.log("Video metadata loaded.");
                if (!canvasRef || !videoRef) return;
                // Set initial canvas size based on video
                canvasRef.width = videoRef.videoWidth;
                canvasRef.height = videoRef.videoHeight;
                console.log(
                    `Canvas size set to: ${canvasRef.width}x${canvasRef.height}`,
                );
                // Now that video is ready, attempt WebSocket connection
                // connectWebSocket(); // Or let the user click the connect button
            };

            console.log("Video stream acquired");
        } catch (err) {
            console.error("Error accessing camera:", err);
            // Handle error - maybe display a message to the user
            alert(
                `Error accessing camera: ${
                    err instanceof Error ? err.message : String(err)
                }. Please ensure permission is granted.`,
            );
        }
    });

    onCleanup(() => {
        console.log("BodyTracker component cleanup");
        isManuallyDisconnected = true; // Ensure no reconnect attempts on cleanup
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
        }
        // Close WebSocket connection
        if (ws) {
            console.log("Closing WebSocket connection during cleanup.");
            ws.onclose = null; // Prevent onclose handler from firing during cleanup
            ws.onerror = null;
            ws.onmessage = null;
            ws.onopen = null;
            ws.close();
            ws = null;
        }
        // Stop video stream tracks
        if (videoRef && videoRef.srcObject) {
            const stream = videoRef.srcObject as MediaStream;
            stream.getTracks().forEach((track) => track.stop());
            videoRef.srcObject = null;
            console.log("Video stream stopped.");
        }
    });

    // --- Render JSX ---
    return (
        <div>
            <div
                id="video-container"
                style={{
                    "text-align": "center",
                    "margin": "auto"
                }}
            >
                {/* Example fixed size */}
                {/* Video element is hidden, canvas shows the feed + overlays */}
                <video
                    ref={videoRef}
                    autoplay
                    playsinline
                    muted // Mute video to allow autoplay in most browsers
                    hidden // Hide the original video feed
                    style={{ width: "320px", height: "320px" }}
                >
                </video>
                {/* Canvas for drawing overlays, positioned over the video */}
                <canvas
                    ref={canvasRef}
                    style={{
                        // position: "absolute",
                        // top: "0",
                        // left: "0",
                        width: "320px", // Make canvas fill the container
                        height: "320px",
                    }}
                >
                </canvas>
            </div>
            <div id="controls" style={{ "margin-top": "10px", "text-align": "center" }}>
                <button
                    ref={connectBtnRef}
                    onClick={connectWebSocket}
                    disabled={isConnected()}
                >
                    Connect
                </button>
                <button
                    ref={disconnectBtnRef}
                    onClick={manualDisconnect}
                    disabled={!isConnected()}
                >
                    Disconnect
                </button>
            </div>
            <div id="results" style={{ "margin-top": "10px", "text-align": "center" }}>
                <div ref={fpsRef}>FPS: {fps()}</div>
                <div ref={squatCountRef}>Squats: {squatCount()}</div>
                <div>
                    Status: {isConnected() ? "Connected" : "Disconnected"}
                </div>
                {/* Display props */}
                <div>Workout: {props.workout}</div>
                <div>User ID: {props.userId}</div>
            </div>
        </div>
    );
};

export default BodyTracker;
