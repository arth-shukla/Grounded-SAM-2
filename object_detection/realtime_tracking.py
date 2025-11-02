import pyzed.sl as sl
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import time
import os
import sys
import datetime
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import torch
import threading
from queue import Queue
from tracking_data_logger import TrackingDataLogger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker

IMAGE_SHAPE = (720, 1280)


class ObjectThrowTracker:
    def __init__(
        self,
        prompt_text,
        detection_interval = 4,
        max_history=300,
        gaussian_sigma=2.0,
        min_speed_threshold=0.5,
        max_object_size=0.1, # in square meters, overestimate
        min_object_size=0.0,
        min_depth=0.3,
        extrapolation_buffer=5,
        record_video=True,
        video_fps=30,
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
        track_rotation=False,
        write_images=False,
    ):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.AUTO
        init_params.camera_fps = video_fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.3
        init_params.depth_maximum_distance = 10.0

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera")
            exit(1)

        # Get camera intrinsic parameters for 3D projection
        camera_info = self.zed.get_camera_information()
        self.fx = camera_info.camera_configuration.calibration_parameters.left_cam.fx
        self.fy = camera_info.camera_configuration.calibration_parameters.left_cam.fy
        self.cx = camera_info.camera_configuration.calibration_parameters.left_cam.cx
        self.cy = camera_info.camera_configuration.calibration_parameters.left_cam.cy
        
        print(f"Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()

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
        self.gaussian_sigma = gaussian_sigma
        self.min_speed_threshold = min_speed_threshold
        self.max_object_size = max_object_size
        self.min_object_size = min_object_size
        self.min_depth = min_depth
        
        self.track_rotation = track_rotation
        
        # Permanent lists to store complete tracking history
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
        self.cameraLastTime = 0.0

        self.processing_times = deque(maxlen=100)
        
        # Video recording settings
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_resolution = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])  # (width, height)
        self.video_writer = None
        self.recording_started = False
        self.write_images = write_images

        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_logger = TrackingDataLogger(log_dir="outputs")
        self.image_buffer, self.image_frame_count = [], []

        # Concurrency: buffer for image, depth, and point cloud
        self.frame_buffer = Queue(maxsize=2)  # Buffer of size 2
        self.buffer_stop_event = threading.Event()
        self.buffer_thread = None

        print("Initialized!")
        print("\n" + "=" * 60)
        self.capture_background()
        print("READY TO TRACK!")
        print(f"Video Recording: {'ENABLED' if self.record_video else 'DISABLED'}")
        print("=" * 60 + "\n")

    def _capture_frames(self):
        """Separate thread that continuously captures image, depth, and point cloud into buffer."""
        while not self.buffer_stop_event.is_set():
            try:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve image
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    frame = self.image.get_data()  # BGRA format
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    
                    # Retrieve depth and point cloud
                    self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
                    depth_mat = sl.Mat()
                    self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    current_depth = depth_mat.get_data()
                    
                    # Pack frame data
                    frame_data = {
                        "frame_bgr": frame_bgr.copy(),
                        "frame_rgb": frame_rgb.copy(),
                        "depth": current_depth.copy(),
                        "timestamp": time.perf_counter()
                    }
                    
                    # Put into buffer (non-blocking, discards oldest if full)
                    try:
                        self.frame_buffer.put_nowait(frame_data)
                    except:
                        # Buffer is full, discard oldest frame
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put_nowait(frame_data)
                        except:
                            pass
            except Exception as e:
                print(f"Error in capture thread: {e}")
                break
        print("Capture thread stopped.")

    def _get_frame_from_buffer(self):
        """Get the latest frame from buffer without blocking."""
        frame_data = None
        try:
            # Try to get the most recent frame without blocking
            while not self.frame_buffer.empty():
                frame_data = self.frame_buffer.get_nowait()
        except:
            pass
        return frame_data

    def get_3d_position(self, x, y):
        """Get XYZ coordinates from depth map."""
        x = round(x)
        y = self.video_resolution[1] - round(y)
        err, point = self.point_cloud.get_value(x, y)
        if err == sl.ERROR_CODE.SUCCESS:
            return np.array([point[0], point[1], point[2]])
        return None
    
    def capture_background(self):
        depth_mat = sl.Mat()
        print("Capturing background reference... Stay clear of the scene!")
        background_frames = []
        for i in range(30):  # Capture 30 frames for stable background
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth = depth_mat.get_data().astype(np.float32)
                depth[depth <= self.min_depth] = np.nan
                depth[depth > 6.0] = np.nan
                background_frames.append(depth)
                print(f"Capturing background frame {i+1}/30")

        # Compute median background (robust to outliers)
        self.background_depth = np.nanmedian(np.array(background_frames), axis=0)
        self.background_depth_mask = ~np.isnan(self.background_depth) & (self.background_depth > self.min_depth) & (self.background_depth < 6.0)
        print("Background captured! You can now enter the scene.")

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

        # Get frame from buffer instead of grabbing directly
        frame_data = self._get_frame_from_buffer()
        
        if frame_data is None:
            return None  # No frame available in buffer
        
        frame_bgr = frame_data["frame_bgr"]
        frame_rgb = frame_data["frame_rgb"]
        current_depth = frame_data["depth"]
        curr_time = time.perf_counter()
        self.cameraLastTime = curr_time
        
        # Store current frame for mouse callback
        self.current_frame = frame_bgr.copy()
        tracked = False

        if self.throw_started:
            current_time = time.time() - self.start_time
            self.frame_count += 1
            print('----------')
            print('Frame:', self.frame_count, 'Camera fps:', self.zed.get_current_fps())
            
            if self.tracking_active and self.frame_count % 2 == 1 and not np.isnan(self.timestamp_buffer[-1]):
                pos_3d, vel_3d = self.linear_extrapolate(current_time, k=3, mode='vel') # more stable
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
                annotated_image = None
            else:
                time1 = time.perf_counter()
                obj, annotated_image = self.detect_thrown_object(frame_rgb, current_depth, current_time, timing=True, sam2_detection_interval=1)
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
                        self.last_obj_dict = {**obj, "velocity_3d": vel_3d}

                        x, y, w, h = bbox
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    pos_3d, vel_3d = self.linear_extrapolate(current_time, k=3, mode='vel') # more stable
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

            if write_images and annotated_image is not None and isinstance(annotated_image, np.ndarray):
                self.image_buffer.append(annotated_image)
                self.image_frame_count.append(self.frame_count)

        status = "TRACKING" if self.tracking_active else (
            "WAITING..." if self.throw_started else "PRESS SPACE TO START"  )
        color = (0, 255, 0) if self.tracking_active else (0, 165, 255) if self.throw_started else (0, 0, 255)
        cv2.putText(frame_bgr, f"Status: {status}", (10, frame_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame_bgr, f"FPS: {self.zed.get_current_fps():.1f}", (frame_bgr.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.fps_history.append(self.zed.get_current_fps())

        if self.record_video and self.video_writer is not None:
            resized = cv2.resize(frame_bgr, self.video_resolution)
            self.video_writer.write(resized)

        return frame_bgr

    def detect_thrown_object(self, frame, current_depth, current_time, timing=False, sam2_detection_interval=-1):
        """Track object using Grounded SAM2 predictor.
        
        Args:
            frame: Input frame to process
            current_depth: Depth map corresponding to the frame
            current_time: Current timestamp
            timing: If True, print timing information for each step
        """
        if timing:
            t_sam = time.perf_counter()

        if current_depth is None:
            print("Warning: No depth data available for tracking.")
            return None, None

        if sam2_detection_interval > 0 and self.tracking_active and self.frame_count % sam2_detection_interval != 0:
            detected = self.background_subtraction_detection(current_depth,
                                                            self.last_obj_dict['bbox'] if self.last_obj_dict else None)
            if not detected:
                print('Background subtraction failed')
                sam2_flag = True
                annotated_image = self.tracker.add_image(frame, full_detection=not self.tracking_active)
            else:
                sam2_flag = False
                print(f'Background subtraction detected {len(detected)} objects')
                # Draw bounding boxes for detected objects
                annotated_image = frame.copy()
                for obj in detected:
                    bbox = obj['bbox']
                    cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        else:
            sam2_flag = True                       
            annotated_image = self.tracker.add_image(frame, full_detection=not self.tracking_active)
                # After we see the object for the first time, disable full detection for next frames
            if annotated_image is None:
                print("Warning: Detection failed, no bounding box found.")
                return None, None
            if timing:
                t_sam_end = time.perf_counter()
                print(f"SAM2 prediction took: {(t_sam_end - t_sam) * 1000:.1f} ms")

        least_displacement = float('inf')
        # If background subtraction was used, find best matching object
        if not sam2_flag:
            best_id, best_position_3d = None, None
            for obj_id, obj in enumerate(detected):
                if obj['depth'] <= 0:
                    print(f"Warning: Invalid depth value {obj['depth']} at centroid.")
                    continue
                area_m2 = obj['area']
                if area_m2 < self.min_object_size or area_m2 > self.max_object_size:
                    print(f'Object with area {area_m2:.4f} m^2 is outside size range.')
                    continue  # Skip objects outside size range
                centroid = obj['centroid']
                position_3d = self.get_3d_position(centroid[0], centroid[1])
                if position_3d is None or np.isnan(position_3d).any():
                    print(f"Warning: Could not get 3D position for object {obj_id} at centroid {centroid}")
                    continue
                if self.last_position is None or self.last_time is None: # Just pick the first valid detection
                    return {**obj, "position_3d": position_3d}, annotated_image
                displacement = np.linalg.norm(position_3d - self.last_position)
                time_diff = current_time - self.last_time
                speed = displacement / (time_diff + 1e-6)
                if speed > 10.0:
                    print(f"Warning: Detected object {obj_id} is moving too fast (speed={speed:.2f} m/s). Skipping.")
                    continue
                if displacement < least_displacement:
                    best_id = obj_id
                    best_position_3d = position_3d
            if best_id is not None:
                best_obj = detected[best_id]
                print('[DEBUG] Detected area m^2:', best_obj['area'])
                return {**best_obj, "position_3d": best_position_3d}, annotated_image
            else:
                print('Warning: No valid object detected by background subtraction')
                return None, None

        best_centroid, best_bbox, best_mask, best_position_3d, best_area = None, None, None, None, None
        print('Number of tracked objects:', len(self.tracker.last_mask_dict.labels))
        for obj_id, obj_info in self.tracker.last_mask_dict.labels.items():
            width_px = max(1, obj_info.x2 - obj_info.x1)
            height_px = max(1, obj_info.y2 - obj_info.y1)

            mask = obj_info.mask
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if xs.numel() == 0:
                # fallback to bbox center if mask empty
                centroid = np.array([obj_info.x1 + width_px / 2.0, obj_info.y1 + height_px / 2.0])
            else:
                centroid = np.array([xs.float().mean().item(), ys.float().mean().item()])

            depth = None
            cx = int(np.clip(centroid[0], 0, current_depth.shape[1] - 1))
            cy = int(np.clip(centroid[1], 0, current_depth.shape[0] - 1))
            depth_value = float(current_depth[cy, cx])
            if depth_value > 0:
                depth = depth_value
                print('[DEBUG] depth at centroid:', depth)
            else:
                print(f"Warning: Invalid depth value {depth_value} at centroid for object {obj_id}.")

            # Use pinhole model to estimate actual 2d size
            if depth is not None:
                if depth < self.min_depth:
                    continue  # Skip objects that are too close
                width_m = (width_px * depth) / self.fx
                height_m = (height_px * depth) / self.fy
                area_m2 = width_m * height_m
                if area_m2 < self.min_object_size or area_m2 > self.max_object_size:
                    print(f'Object {obj_id} with area {area_m2:.4f} m^2 is outside size range.')
                    continue  # Skip objects outside size range
            else:
                print(f"Warning: No valid depth at object centroid for object {obj_id}.")
                continue
            
            position_3d = self.get_3d_position(centroid[0], centroid[1])
            if position_3d is None or np.isnan(position_3d).any():
                print(f"Warning: Could not get 3D position for object {obj_id} at centroid {centroid}")
                continue
            if self.last_position is None or self.last_time is None: # Just pick the first valid detection
                return {
                    "centroid": centroid,
                    "bbox": (obj_info.x1, obj_info.y1, width_px, height_px),
                    "mask": obj_info.mask,
                    "area": area_m2,
                    "position_3d": position_3d
                }, annotated_image
            displacement = np.linalg.norm(position_3d - self.last_position)
            time_diff = current_time - self.last_time
            speed = displacement / (time_diff + 1e-6)
            if speed > 10.0:
                print(f"Warning: Detected object {obj_id} is moving too fast (speed={speed:.2f} m/s). Skipping.")
                continue
            if displacement < least_displacement:
                least_displacement = displacement
                best_mask = obj_info.mask
                best_centroid = centroid
                best_bbox = (obj_info.x1, obj_info.y1, width_px, height_px)
                best_area = area_m2 if depth is not None else None
                best_position_3d = position_3d
        if best_area is not None:
            print('[DEBUG] Best area m^2:', best_area)
        if best_centroid is None:
            print('Warning: Object not detected by Grounded SAM2')
            return None, None
        return {
            "centroid": best_centroid,
            "bbox": best_bbox,
            "mask": best_mask,
            "area": best_area,
            "position_3d": best_position_3d
        }, annotated_image
    
    
    def background_subtraction_detection(self, current_depth, prev_bbox):
        def inflated_crop(inflation=2.0):
            assert prev_bbox is not None, "Previous bounding box is required for inflation."
            x, y, w, h = prev_bbox
            new_x = np.clip(x + (1 - inflation) * (w // 2), 0, current_depth.shape[1] - w * inflation)
            new_y = np.clip(y + (1 - inflation) * (h // 2), 0, current_depth.shape[0] - h * inflation)
            inflated = (int(new_x), 
                        int(new_y), 
                        int(w * inflation), 
                        int(h * inflation))
            return inflated
        
        # Gaussian smoothing
        current_depth_mask = ~np.isnan(current_depth) & (current_depth > self.min_depth) & (current_depth < 6.0)
        current_depth_mask = current_depth_mask & self.background_depth_mask
        depth_diff = np.abs(self.background_depth - current_depth)
        # smoothed = cv2.GaussianBlur(current_depth, (5, 5), sigmaX=1.5)
        cropping_box = inflated_crop(inflation=3.0)
        if cropping_box is not None and cropping_box[2] > 20 and cropping_box[3] > 20:
            x, y, w, h = cropping_box
            depth_diff = depth_diff[y:y+h, x:x+w]
            # Foreground: objects significantly closer than background
            foreground_mask = ((depth_diff > 0.1) & current_depth_mask[y:y+h, x:x+w]).astype(np.uint8) * 255  # 100mm threshold
        # print(f'[DEBUG] Foreground mask stats: min={foreground_mask.min()}, max={foreground_mask.max()}, nonzero={np.count_nonzero(foreground_mask)}')
        # Morphological operations to clean up mask
        kernel_open = np.ones((5, 5), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Then close to fill gaps within the object
        kernel_close = np.ones((7, 7), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_close)

        # Return to binary mask (0 or 1)
        foreground_mask = (foreground_mask > 50).astype(np.uint8)
        # DEBUG
        # filename = os.path.join('outputs', self.datetime_str, 'foreground', f'foreground_mask_{self.frame_count}.png')
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # cv2.imwrite(filename, (foreground_mask * 255).astype(np.uint8))

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground_mask, connectivity=8
        )

        current_detections = []  # Store all detections this frame

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            # Calculate centroid
            cX, cY = int(centroids[i][0]), int(centroids[i][1])
                
            # Check if within valid image bounds
            if 0 <= cY < depth_diff.shape[0] and 0 <= cX < depth_diff.shape[1]:
                depth_val = current_depth[cY, cX]
                print('[DEBUG] Detected component depth:', depth_val)
                
                # Only track if depth is valid and in reasonable range
                if not np.isnan(depth_val) and self.min_depth < depth_val < 6.0:
                    # Filter by size
                    actual_area = (area * (depth_val ** 2)) / (self.fx * self.fy)
                    if self.min_object_size <= actual_area <= self.max_object_size:
                        # Compute bounding box
                        x1, y1, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                        mask = (labels == i).astype(np.uint8)
                        print('[DEBUG] Cropped bbox:', cropping_box)
                        print('[DEBUG] Original bbox:', (x1, y1, w, h))
                        bbox = (x1 + cropping_box[0], y1 + cropping_box[1], w, h)
                        centroid = np.array([cX + cropping_box[0], cY + cropping_box[1]])
                        mask_cropped = mask.copy()
                        mask = np.zeros_like(current_depth, dtype=np.uint8)
                        mask[cropping_box[1]:cropping_box[1] + cropping_box[3], 
                                cropping_box[0]:cropping_box[0] + cropping_box[2]] = mask_cropped
                        current_detections.append({
                            "centroid": centroid,
                            "area": actual_area,
                            "depth": depth_val,
                            "mask": mask,
                            "bbox": bbox
                        })
        return current_detections

    def run(self):
        """Main loop."""
        print("\nControls:")
        print("  SPACE       - Start tracking mode")
        print("  S           - Start/Stop recording")
        # print("  R           - Reset background")
        print("  P           - Generate plots")
        print("  Q           - Quit")
        
        # Start the capture thread
        self.buffer_stop_event.clear()
        self.buffer_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.buffer_thread.start()
        print("Capture thread started.")
        
        # Give the buffer some time to fill
        time.sleep(0.1)
        
        while True:
            if self.record_video and self.throw_started and not self.recording_started:
                self.start_recording()
            time1 = time.perf_counter()
            frame = self.track_frame(write_images=self.write_images)
            time2 = time.perf_counter()
            print(f"frame processing time: {(time2 - time1) * 1000:.3f} ms")
            if frame is not None:
                if self.recording_started:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame first to create the window
                cv2.imshow("Object Throw Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                
                # Stop the capture thread
                self.buffer_stop_event.set()
                if self.buffer_thread is not None:
                    self.buffer_thread.join(timeout=2.0)
                    print("Capture thread stopped.")
                
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

        cv2.destroyAllWindows()
        self.zed.close()

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
    
    parser = argparse.ArgumentParser(description='Track objects using SAM2 with ZED camera')
    parser.add_argument('prompt', type=str, help='Text prompt for object detection')
    parser.add_argument('--detection-interval', type=int, default=4, help='Detection interval for SAM2')
    parser.add_argument('--max-history', type=int, default=300, help='Maximum number of frames to keep in history')
    parser.add_argument('--min-speed-threshold', type=float, default=0.5, help='Minimum speed threshold for tracking')
    parser.add_argument('--max-object-size', type=float, default=0.1, help='Maximum object size (square meters)')
    parser.add_argument('--min-object-size', type=float, default=1e-3, help='Minimum object size (square meters)')
    parser.add_argument('--min-depth', type=float, default=0.3, help='Minimum depth for detection (meters)')
    parser.add_argument('--record', action='store_true', help='Enable video recording')
    parser.add_argument('--video-fps', type=int, default=30, help='Video recording FPS')
    parser.add_argument('--sam2-config', type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml", help='Path to SAM2 config file')
    parser.add_argument('--sam2-checkpoint', type=str, default="checkpoints/sam2.1_hiera_tiny.pt", help='Path to SAM2 checkpoint file')
    parser.add_argument('--write-images', action='store_true', help='Save annotated images')
    args = parser.parse_args()

    tracker = ObjectThrowTracker(
        prompt_text=args.prompt,
        detection_interval=args.detection_interval,
        max_history=args.max_history,
        min_speed_threshold=args.min_speed_threshold,
        max_object_size=args.max_object_size,
        min_object_size=args.min_object_size,
        min_depth=args.min_depth,
        record_video=args.record,
        video_fps=args.video_fps,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        write_images=args.write_images,
    )
    tracker.run()
