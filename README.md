create a Astro + SolidJs app with SSR site at /[workout]/[user_id]

which connects to websocket url and has a video html which records the camera and sends frames only at 24 fps to the backend 

the response structure is like this 

{
                        "keypoints": [[int, int]; 17],
                        "confidence": [float; 17],
                        # "orig_shape": orig_shape.tolist(),
                        "frame_count": int,
                        "squat_count": int,
                        "plank_count": int,
                        "pushup_count": int,
                        "lunges_count": int,
                        "jumping_jacks_count": int,
                        "squat_bool": bool,
                        "lunges_bool": bool,
                        "plank_bool": bool,
                        "pushup_bool": bool,
                        "jumping_jacks_bool": bool
                    }

depending on the workout in url path 
send the data to firebase table with columns [user_id, workout, count]

use the bool value  to show tick or cross sign in website to signify that the user is doing in correct posture

and use this response to draw the overlay of human skeleton with points and lines on the video feed 

you can use tailwind for styling and only use the latest versions of the libraries