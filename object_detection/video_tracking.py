import cv2
import numpy as np
from collections import deque
import time
import os
import sys
import datetime
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import torch
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker
from tracking_data_logger import TrackingDataLogger

COLOR_DIST_THRESHOLD = 0.03

# Camera intrinsics for ZED camera (adjust these based on your camera)
CAMERA_INTRINSICS = {
    'fx': 1058.9,
    'fy': 1058.9,
    'cx': 964.5,
    'cy': 533.7,
}


class ObjectThrowTracker:
    def __init__(
        self,
        rgb_video_path,
        xyz_data_path,
        depth_data_path,
        prompt_text,
        detection_interval=4,
        max_history=300,
        min_speed_threshold=0.5,
        max_object_size=0.1,  # in square meters, overestimate
        min_object_size=0.0,
        min_depth=0.3,
        extrapolation_buffer=5,
        video_fps=60, 
        record_video=True,
        write_images=True,
        video_resolution=(1280, 720),
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
        track_rotation=False
    ):
        # Open RGB video file and load depth data
        print(f"Opening RGB video file: {rgb_video_path}")
        self.cap_rgb = cv2.VideoCapture(str(rgb_video_path))  # Ensure string path
        
        # More detailed error checking for video opening
        if not self.cap_rgb.isOpened():
            print(f"Failed to open RGB video file: {rgb_video_path}")
            print("Video properties:")
            print(f"  Exists: {os.path.exists(rgb_video_path)}")
            print(f"  Size: {os.path.getsize(rgb_video_path) if os.path.exists(rgb_video_path) else 'N/A'} bytes")
            print(f"  Absolute path: {os.path.abspath(rgb_video_path)}")
            exit(1)

        # Get video properties first to verify the file is readable
        self.video_fps = int(self.cap_rgb.get(cv2.CAP_PROP_FPS))
        self.video_width = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Successfully opened RGB video with properties:")
        print(f"  FPS: {self.video_fps}")
        print(f"  Width: {self.video_width}")
        print(f"  Height: {self.video_height}")
        print(f"  Total frames: {self.total_frames}")

        # Test reading the first frame
        ret, test_frame = self.cap_rgb.read()
        if not ret or test_frame is None:
            print("Error: Could not read the first frame from the video")
            exit(1)
        # Reset video to start
        self.cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Load point cloud data from .npy file
        print(f"Loading point cloud data from: {xyz_data_path}")
        try:
            self.point_cloud = np.load(xyz_data_path)
            print(f"Loaded point cloud data shape: {self.point_cloud.shape}")
        except Exception as e:
            print(f"Failed to load point cloud data from {xyz_data_path}: {e}")
            exit(1)
        
        # Verify point cloud data dimensions match video (frames, height, width, xyz)
        if (self.point_cloud.shape != (self.total_frames, self.video_height, self.video_width, 4)):
            print(f"Error: Point cloud data shape {self.point_cloud.shape} doesn't match expected dimensions "
                  f"({self.total_frames}, {self.video_height}, {self.video_width}, 4)")
            exit(1)
        
        # Load depth data from .npy file
        print(f"Loading depth data from: {depth_data_path}")
        try:
            self.depth_map = np.load(depth_data_path)
            print(f"Loaded depth data shape: {self.depth_map.shape}")
        except Exception as e:
            print(f"Failed to load point cloud data from {depth_data_path}: {e}")
            exit(1)

        print(f"Video properties:")
        print(f"  Resolution: {self.video_width}x{self.video_height}")
        print(f"  FPS: {self.video_fps}")
        print(f"  Total frames: {self.total_frames}")

        # Camera intrinsics for 3D projection
        self.fx = CAMERA_INTRINSICS['fx']
        self.fy = CAMERA_INTRINSICS['fy']
        self.cx = CAMERA_INTRINSICS['cx']
        self.cy = CAMERA_INTRINSICS['cy']
        
        print(f"Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

        # SAM2 Configuration
        self.tracker = IncrementalObjectTracker(
            grounding_model_id="IDEA-Research/grounding-dino-tiny",
            sam2_model_cfg=sam2_config,
            sam2_ckpt_path=os.path.join(_PROJECT_ROOT, sam2_checkpoint),
            device="cuda",
            prompt_text=prompt_text,
            detection_interval=detection_interval,
        )
        self.tracker.set_prompt(prompt_text)

        self.max_history = max_history
        self.min_speed_threshold = min_speed_threshold
        self.max_object_size = max_object_size
        self.min_object_size = min_object_size
        self.min_depth = min_depth
        self.track_rotation = track_rotation
        
        # Lists to store tracking history
        self.positions_history = []
        self.velocities_history = []
        self.timestamps_history = []
        self.frame_numbers_history = []
        self.fps_history = []

        # extrapolation buffer
        self.position_buffer = deque(maxlen=extrapolation_buffer)
        self.velocity_buffer = deque(maxlen=extrapolation_buffer)
        self.timestamp_buffer = deque(maxlen=extrapolation_buffer)

        self.last_position = None
        self.last_time = None
        self.last_obj_dict = None
        self.start_time = time.time()
        self.frame_count = 0
        self.tracking_active = False
        self.throw_started = False
        self.lastFrameTime = 0.0

        self.processing_times = deque(maxlen=100)
        
        # Video recording settings
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_resolution = video_resolution
        self.video_writer = None
        self.recording_started = False
        self.write_images = write_images
        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_logger = TrackingDataLogger(log_dir="outputs")
        self.image_buffer, self.image_frame_count = [], []

        print("Initialized!")
        
        print("\n" + "=" * 60)
        print("READY TO TRACK!")
        print(f"Video Recording: {'ENABLED' if self.record_video else 'DISABLED'}")
        print("=" * 60 + "\n")

    def get_3d_position(self, x, y, pc):
        """Get XYZ coordinates from depth map."""
        x = round(x)
        y = round(y)
        point = pc[y, x, :3]
        if np.isnan(point).any():
            print("Warning: 3D point contains NaN values")
            return None
        err = pc[y, x, 3] # not used because we assume we kept valid points only
        return point

    def calculate_velocity(self, position, current_time):
        """Estimate velocity from position delta over time."""
        if self.last_position is None or self.last_time is None:
            return np.array([0.0, 0.0, 0.0])
        dt = current_time - self.last_time
        if dt == 0:
            return np.array([0.0, 0.0, 0.0])
        return (position - self.last_position) / dt
    
    def linear_extrapolate(self, current_time, k=3, mode='acc'):
        """Linearly extrapolate position based on last k positions."""
        # Find the index of the first self.frame_numbers_history that is non-smaller than self.frame_count - k
        if len(self.position_buffer) < k:
            return None, None
        valid_indices = np.argwhere(~np.isnan(np.array(self.position_buffer)).any(1)).flatten()
        n_valid = len(valid_indices)
        if n_valid < k:
            return None, None
        pos_array = np.array(self.position_buffer)[valid_indices]
        vel_array = np.array(self.velocity_buffer)[valid_indices]
        time_array = np.array(self.timestamp_buffer)[valid_indices]
        if mode == 'acc':
            estimated_acc = (np.mean(vel_array[1:], axis=0) - np.mean(vel_array[:-1], axis=0)) / (time_array[-1] - time_array[0] + 1e-6)
            time_shift = current_time - time_array[-1]
            extrapolated_vel = time_shift * estimated_acc + vel_array[-1]
            extrapolated_pos = pos_array[-1] + extrapolated_vel * time_shift
        else:  # mode == 'vel'
            extrapolated_vel = np.mean(vel_array, axis=0)
            time_shift = current_time - time_array[-1]
            extrapolated_pos = pos_array[-1] + extrapolated_vel * time_shift
        return extrapolated_pos, extrapolated_vel
    
    def track_frame(self, write_images=False):
        """Process a single frame and track the object."""
        frame_start = time.perf_counter()

        # Read frame (in BGR format)
        ret_rgb, frame_bgr = self.cap_rgb.read()
        if not ret_rgb or frame_bgr is None:
            print(f"Failed to read frame at position {self.frame_count}")
            print(f"Video state:")
            print(f"  Current position: {int(self.cap_rgb.get(cv2.CAP_PROP_POS_FRAMES))}")
            print(f"  Total frames: {self.total_frames}")
            print(f"  Is opened: {self.cap_rgb.isOpened()}")
            
            # Try to recover by seeking to the frame
            if self.cap_rgb.isOpened():
                print(f"Attempting to seek to frame {self.frame_count}")
                self.cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                ret_rgb, frame_bgr = self.cap_rgb.read()
                if not ret_rgb or frame_bgr is None:
                    print("Recovery attempt failed")
                    return False, None
                print("Successfully recovered frame")
            else:
                return False, None
                
        # Convert BGR to RGB for processing
        # frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Get corresponding point cloud frame from numpy array
        if self.frame_count >= self.total_frames:
            print(f"End of frames reached: {self.frame_count} >= {self.total_frames}")
            return False, None
            
        xyz_frame = self.point_cloud[self.frame_count].astype(np.float32)
        current_depth = self.depth_map[self.frame_count].astype(np.float32)
        curr_time = time.perf_counter()
        print('Time since last frame:', (curr_time - self.lastFrameTime) * 1000, 'ms')
        self.lastFrameTime = curr_time
        
        self.current_frame = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = False

        current_time = time.time() - self.start_time
        self.frame_count += 1
        print('----------')
        print(f'Frame: {self.frame_count}/{self.total_frames}, FPS: {self.video_fps}')
        
        time1 = time.perf_counter()
        obj, annotated_image = self.detect_thrown_object(frame_rgb, current_depth, xyz_frame, timing=False)
        print(f"detect_thrown_object used: {(time.perf_counter() - time1) * 1000:.3f} ms")
        
        if obj:
            bbox = obj["bbox"]
            cx, cy = obj["centroid"].astype(int)
            pos_3d = obj['position_3d']
            if pos_3d is not None:
                vel_3d = self.calculate_velocity(pos_3d, current_time)
                print('Detected position:', pos_3d)
                print('Detected velocity:', vel_3d)
                if not self.tracking_active:
                    self.tracking_active = True
                
                self.positions_history.append(pos_3d)
                self.velocities_history.append(vel_3d)
                self.timestamps_history.append(current_time)
                self.frame_numbers_history.append(self.frame_count)

                self.position_buffer.append(pos_3d)
                self.velocity_buffer.append(vel_3d)
                self.timestamp_buffer.append(current_time)

                self.last_position = pos_3d
                self.last_time = current_time
                self.last_obj_dict = obj
                tracked = True

                x, y, w, h = bbox
                cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(self.current_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw 3D position text
                pos_text = f"3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})m"
                cv2.putText(self.current_frame, pos_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            pos_3d, vel_3d = self.linear_extrapolate(current_time)
            if pos_3d is not None and vel_3d is not None:
                print('Extrapolated position:', pos_3d)
                print('Extrapolated velocity:', vel_3d)
                self.positions_history.append(pos_3d)
                self.velocities_history.append(vel_3d)
                self.timestamps_history.append(current_time)
                self.frame_numbers_history.append(self.frame_count)
            self.position_buffer.append(np.array([np.nan, np.nan, np.nan]))
            self.velocity_buffer.append(np.array([np.nan, np.nan, np.nan]))
            self.timestamp_buffer.append(np.nan)
        if annotated_image is not None and isinstance(annotated_image, np.ndarray):
            self.image_buffer.append(annotated_image)
            self.image_frame_count.append(self.frame_count)

        status = "TRACKING" if self.tracking_active else "PRESS SPACE TO START"
        color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
        cv2.putText(self.current_frame, f"Status: {status}", (10, self.current_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_number_text = f"Frame: {self.frame_count}/{self.total_frames}"
        cv2.putText(self.current_frame, frame_number_text, (self.current_frame.shape[1] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.fps_history.append(self.video_fps)

        if self.record_video and self.video_writer is not None:
            resized = cv2.resize(self.current_frame, self.video_resolution)
            self.video_writer.write(resized)

        if write_images and annotated_image is not None and isinstance(annotated_image, np.ndarray):          
            self.image_buffer.append(annotated_image)
            self.image_frame_count.append(self.frame_count)

        return tracked, self.current_frame
    
    def detect_thrown_object(self, frame, current_depth, xyz_frame, timing=False):
        """Track object using Grounded SAM2 predictor.
        
        Args:
            frame: Input RGB frame to process
            xyz_frame: Corresponding point cloud frame (XYZ coordinates in meters)
            timing: If True, print timing information for each step
        """
        if timing:
            t_sam = time.perf_counter()
            
        annotated_image = self.tracker.add_image(frame, full_detection=not self.tracking_active)
        if annotated_image is None:
            print("Warning: Detection failed, no bounding box found.")
            return None, None
        
        if timing:
            t_sam_end = time.perf_counter()
            print(f"SAM2 prediction took: {(t_sam_end - t_sam) * 1000:.1f} ms")

        least_displacement = float('inf')
        best_centroid, best_bbox, best_mask, best_area, best_position_3d = None, None, None, None, None
        print('Number of tracked objects:', len(self.tracker.last_mask_dict.labels))
        
        for obj_id, obj_info in self.tracker.last_mask_dict.labels.items():
            width_px = max(1, obj_info.x2 - obj_info.x1)
            height_px = max(1, obj_info.y2 - obj_info.y1)

            centroid = np.array([obj_info.x1 + obj_info.x2, obj_info.y1 + obj_info.y2]) / 2

            # Get 3D position directly from point cloud
            position_3d = self.get_3d_position(centroid[0], centroid[1], xyz_frame)
            if position_3d is None:
                print(f"Warning: No valid 3D position for object {obj_id}")
                continue

            # Check minimum depth constraint using Z coordinate
            depth = current_depth[int(centroid[1]), int(centroid[0])]
            if np.isnan(depth) or depth < self.min_depth:
                print(f"Object {obj_id} has invalid depth: {depth}")
                continue
            depth = np.clip(depth, 0.0, 10.0)
            print(f'[DEBUG] depth: {depth} m')

            # Calculate object size in meters
            width_m = (width_px * depth) / self.fx
            height_m = (height_px * depth) / self.fy
            area_m2 = width_m * height_m

            if area_m2 < self.min_object_size or area_m2 > self.max_object_size:
                print(f'Object {obj_id} with area {area_m2:.4f} m^2 is outside size range.')
                continue
            if position_3d is None:
                print(f"Warning: Could not get 3D position for object {obj_id} at centroid {centroid}")
                continue

            # For first detection, just pick the first valid object
            if self.last_position is None or np.isnan(self.last_position).any():
                return {
                    "centroid": centroid,
                    "bbox": (obj_info.x1, obj_info.y1, width_px, height_px),
                    "mask": obj_info.mask,
                    "area": area_m2,
                    "depth": depth,
                    "position_3d": position_3d
                }, annotated_image

            # Pick the object closest to the last detected position
            displacement = np.linalg.norm(position_3d - self.last_position)
            if displacement < least_displacement:
                least_displacement = displacement
                best_mask = obj_info.mask
                best_centroid = centroid
                best_bbox = (obj_info.x1, obj_info.y1, width_px, height_px)
                best_area = area_m2
                best_depth = depth
                best_position_3d = position_3d

        if best_centroid is None:
            print('Warning: No valid objects detected')
            return None, None
        if best_area is not None:
            print(f'Best object - Area: {best_area:.4f} m²')
        return {
            "centroid": best_centroid,
            "bbox": best_bbox,
            "mask": best_mask,
            "area": best_area,
            "depth": best_depth,
            "position_3d": best_position_3d
        }, annotated_image

    def run(self):
        """Main loop."""
        print("\nControls:")
        print("  SPACE       - Start tracking mode")
        print("  S           - Start/Stop recording")
        print("  P           - Generate plots")
        print("  Q           - Quit")

        # Store RGB path for potential reopening
        try:
            self.rgb_path = str(self.cap_rgb.get(cv2.CAP_PROP_FILENAME))
            print(f"Video file path: {self.rgb_path}")
        except Exception as e:
            print(f"Warning: Could not get video filename: {e}")
            self.rgb_path = None
        
        end_of_video = False
        while True:
            try:
                # Verify video is still opened
                if not self.cap_rgb.isOpened():
                    print("Error: Video file was closed unexpectedly")
                    if self.rgb_path:
                        print("Attempting to reopen...")
                        self.cap_rgb.release()
                        self.cap_rgb = cv2.VideoCapture(self.rgb_path)
                        if not self.cap_rgb.isOpened():
                            print("Failed to reopen video file")
                            return
                    else:
                        print("Cannot reopen: no file path stored")
                        return

                if self.record_video and self.throw_started and not self.recording_started:
                    self.start_recording()
                time1 = time.perf_counter()
                tracked, frame = self.track_frame(write_images=self.write_images)
                time2 = time.perf_counter()
                print(f"frame processing time: {(time2 - time1) * 1000:.3f} ms")

                if frame is None:
                    print("End of video or error reading frame")
                    end_of_video = True

            except Exception as e:
                print(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                break

            if frame is not None:
                if self.recording_started:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame first to create the window
                cv2.imshow("Object Throw Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or end_of_video:
                print("\nQuitting...")
                if self.recording_started:
                    self.stop_recording()
                
                if self.positions_history and self.velocities_history:
                    print("Saving tracking data...")
                    self.data_logger.save_tracking_data(self)
                    print("Generating plots...")
                    self.data_logger.plot_tracking_data(self, save_plots=False)
                if self.image_buffer:
                    print("Saving annotated images...")
                    self.data_logger.save_annotated_images(self)
                break
            elif key == ord(" ") and not self.throw_started:
                self.throw_started = True
                print("\nTracking started...")
            elif key == ord("s"):
                if self.recording_started:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord("p"):
                if self.positions_history and self.velocities_history:
                    print("\nGenerating plots...")
                    self.data_logger.plot_tracking_data(self, save_plots=True)
                else:
                    print("\nNo tracking data available to plot yet.")

        # Clean up
        cv2.destroyAllWindows()
        self.cap_rgb.release()
        if self.video_writer is not None:
            self.video_writer.release()

    def start_recording(self, filename="outputs/videos/zed_video.mp4"):
        """Start video recording."""
        if not self.record_video:
            print("Video recording is disabled.")
            return
        
        if self.video_writer is not None:
            print("Recording already in progress.")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, self.video_fps, self.video_resolution
        )
        self.recording_started = True
        print(f"\n🔴 Recording started: {filename}")
        print(f"   Resolution: {self.video_resolution}")
        print(f"   FPS: {self.video_fps}\n")

    def stop_recording(self):
        """Stop video recording and save file."""
        if self.video_writer is not None:
            self.video_writer.release()
            print("\n✓ Video saved successfully!")
            self.video_writer = None
            self.recording_started = False
        else:
            print("No recording in progress.")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Track objects in video using SAM2 with depth information')
    parser.add_argument('video_dir', type=str, help='Path to directory containing rgb.avi and depth.avi')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for object detection')
    parser.add_argument('--detection-interval', type=int, default=4, help='Interval between detections')
    parser.add_argument('--min-depth', type=float, default=0.3, help='Minimum depth for detection (meters)')
    parser.add_argument('--max-object-size', type=float, default=0.1, help='Maximum object size (square meters)')
    parser.add_argument('--min-object-size', type=float, default=1e-3, help='Minimum object size (square meters)')
    parser.add_argument('--record', action='store_true', help='Enable video recording')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    rgb_path = video_dir / "rgb.avi"
    xyz_path = video_dir / "xyz.npy"
    depth_path = video_dir / "depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB video file not found: {rgb_path}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"Point cloud data file not found: {xyz_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth data file not found: {depth_path}")

    tracker = ObjectThrowTracker(
        rgb_video_path=str(rgb_path),
        xyz_data_path=str(xyz_path),
        depth_data_path=str(depth_path),
        prompt_text=args.prompt,
        detection_interval=args.detection_interval,
        min_depth=args.min_depth,
        max_object_size=args.max_object_size,
        min_object_size=args.min_object_size,
        record_video=args.record,
        video_resolution=(640, 720),
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
    )
    tracker.run()
