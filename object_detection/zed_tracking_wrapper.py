import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import threading
from queue import Queue

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pyzed.sl as sl

from object_detection.object_throw_tracker_array import ObjectThrowTracker


def open_zed(fps=30):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.3
    init_params.depth_maximum_distance = 10.0

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Failed to open ZED camera")
    return zed


def get_zed_intrinsics(zed):
    cam_info = zed.get_camera_information()
    fx = cam_info.camera_configuration.calibration_parameters.left_cam.fx
    fy = cam_info.camera_configuration.calibration_parameters.left_cam.fy
    cx = cam_info.camera_configuration.calibration_parameters.left_cam.cx
    cy = cam_info.camera_configuration.calibration_parameters.left_cam.cy
    return fx, fy, cx, cy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Text prompt for object detection")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", type=str, default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--write-images", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Open ZED camera
    zed = open_zed(fps=args.video_fps)
    fx, fy, cx, cy = get_zed_intrinsics(zed)
    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Create tracker bound to arrays, supply intrinsics
    tracker = ObjectThrowTracker(
        prompt_text=args.prompt,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        record_video=args.record,
        video_fps=args.video_fps,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
    )

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth_mat = sl.Mat()

    # Frame buffer and capture thread (size 2) to decouple grabbing from processing
    frame_buffer: "Queue[dict]" = Queue(maxsize=2)
    buffer_stop_event = threading.Event()

    def capture_thread():
        while not buffer_stop_event.is_set():
            try:
                if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()  # BGRA
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    depth = depth_mat.get_data()

                    frame_data = {
                        "frame_rgb": frame_rgb.copy(),
                        "depth": depth.copy(),
                        "timestamp": time.perf_counter(),
                    }
                    try:
                        frame_buffer.put_nowait(frame_data)
                    except Exception:
                        # buffer full -> discard oldest then put
                        try:
                            frame_buffer.get_nowait()
                            frame_buffer.put_nowait(frame_data)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Capture thread error: {e}")
                break

    def get_latest_frame():
        latest = None
        try:
            while not frame_buffer.empty():
                latest = frame_buffer.get_nowait()
        except Exception:
            pass
        return latest

    print("Controls:\n  SPACE - Start tracking\n  S - Start/Stop recording\n  Q - Quit")

    # Start capture thread
    buffer_stop_event.clear()
    buf_thread = threading.Thread(target=capture_thread, daemon=True)
    buf_thread.start()

    try:
        while True:
            frame_data = get_latest_frame()
            if frame_data is None:
                # nothing available yet
                time.sleep(0.005)
                continue

            frame_rgb = frame_data["frame_rgb"]
            depth = frame_data["depth"]
            timestamp = frame_data.get("timestamp", time.perf_counter())

            frame_bgr, obj, annotated = tracker.process_frame(frame_rgb, depth, timestamp=timestamp, write_images=args.write_images)
            if frame_bgr is None:
                # nothing to display
                continue

            # Overlay status text and FPS similar to realtime_tracking_kalman
            status = "TRACKING" if tracker.tracking_active else ("WAITING..." if tracker.throw_started else "PRESS SPACE TO START")
            color = (0, 255, 0) if tracker.tracking_active else (0, 165, 255) if tracker.throw_started else (0, 0, 255)
            cv2.putText(frame_bgr, f"Status: {status}", (10, frame_bgr.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Use ZED-reported FPS when available
            try:
                fps_val = zed.get_current_fps()
            except Exception:
                fps_val = 0.0
            cv2.putText(frame_bgr, f"FPS: {fps_val:.1f}", (frame_bgr.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            tracker.fps_history.append(fps_val)

            # Draw bounding box only when object detected
            if obj is not None:
                try:
                    x, y, w, h = obj.get("bbox", (0, 0, 0, 0))
                    cx, cy = obj.get("centroid", (0, 0))
                    cv2.rectangle(frame_bgr, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
                    cv2.circle(frame_bgr, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                except Exception:
                    pass

            # Recording indicator
            if tracker.recording_started:
                cv2.circle(frame_bgr, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame_bgr, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show frame
            cv2.imshow("Object Throw Tracker (ZED)", frame_bgr)

            # Write overlayed frame to video if recording (delegate writing to wrapper so overlay is included)
            if tracker.record_video and tracker.video_writer is not None:
                try:
                    resized = cv2.resize(frame_bgr, tracker.video_resolution)
                    tracker.video_writer.write(resized)
                except Exception as e:
                    print(f"Error writing video frame: {e}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord(" "):
                if not tracker.throw_started:
                    tracker.throw_started = True
                    tracker.start_time = time.perf_counter()
                    print("Tracking started")
            elif key == ord("s"):
                if tracker.recording_started:
                    tracker.stop_recording()
                    print("Stopped recording")
                else:
                    tracker.start_recording()
                    print("Started recording")
    finally:
        # Stop capture thread and cleanup
        buffer_stop_event.set()
        buf_thread.join(timeout=2.0)

    # Cleanup
    cv2.destroyAllWindows()
    # Delegate finalization to the tracker (stops recording and saves data)
    try:
        tracker.finalize(save_plots=args.write_images)
    except Exception as e:
        print(f"Error during tracker finalization: {e}")

    zed.close()


if __name__ == "__main__":
    main()
