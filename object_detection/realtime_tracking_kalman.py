import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker
from utils.kalman_filter import EKF
from .tracking_data_logger import TrackingDataLogger

IMAGE_SHAPE = (720, 1280)  # (h, w)


class ObjectThrowTracker:
    """Array-based variant of ObjectThrowTracker.

    Accepts RGB and depth as numpy arrays per-frame via process_frame().

    Parameters (important ones):
    - prompt_text: str prompt for Grounded SAM2
    - sam2_config, sam2_checkpoint: paths for SAM2
    - fx, fy, cx, cy: optional camera intrinsics. If None, sensible defaults used based on frame size.
    """

    def __init__(
        self,
        prompt_text,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        max_history=300,
        gaussian_sigma=2.0,
        min_speed_threshold=0.5,
        max_object_size=0.1,
        min_object_size=0.0,
        min_depth=0.3,
        record_video=True,
        video_fps=30,
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
        track_rotation=False,
        write_images=False,
        device="cuda",
    ):
        # SAM2 tracker
        self.tracker = IncrementalObjectTracker(
            grounding_model_id="IDEA-Research/grounding-dino-tiny",
            sam2_model_cfg=sam2_config,
            sam2_ckpt_path=os.path.join(_PROJECT_ROOT, sam2_checkpoint),
            device=device,
            prompt_text=prompt_text,
        )
        self.tracker.set_prompt(prompt_text)

        # Camera intrinsics default (will be set on first frame if None)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.max_history = max_history
        self.gaussian_sigma = gaussian_sigma
        self.min_speed_threshold = min_speed_threshold
        self.max_object_size = max_object_size
        self.min_object_size = min_object_size
        self.min_depth = min_depth
        self.track_rotation = track_rotation

        # Kalman filter
        self.kalman = EKF(
            init_pos_std=0.01,
            init_vel_std=0.1,
            pos_process_std=0.01,
            vel_process_std_xy=0.5,
            vel_process_std_z=0.5,
            meas_std=0.1,
            k=0.05,
        )
        self.kalman_active = False
        self.min_motion_detections = 3
        self.motion_threshold = 0.5
        self.motion_count = 0
        self.kalman_active_frame = 0
        self.initial_vel = np.zeros(3)
        self.stagnating_count = 0

        # Histories
        self.positions_history = []
        self.velocities_history = []
        self.timestamps_history = []
        self.frame_numbers_history = []
        self.fps_history = []

        self.last_position = None
        self.last_detected_time = None
        self.last_time = None
        self.last_obj_dict = None
        self.frame_count = 0
        self.tracking_active = False
        self.throw_started = False
        self.start_time = None

        self.processing_times = deque(maxlen=100)

        # Video recording
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_resolution = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
        self.video_writer = None
        self.recording_started = False
        self.write_images = write_images

        self.datetime_str = time.strftime("%Y%m%d_%H%M%S")
        self.data_logger = TrackingDataLogger(log_dir="outputs")
        self.image_buffer = []
        self.image_frame_count = []

        print("ArrayObjectThrowTracker initialized")

    def _ensure_intrinsics(self, frame):
        h, w = frame.shape[:2]
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            # Reasonable defaults: center principal point and focal ~0.8*width
            self.fx = self.fx or (0.8 * w)
            self.fy = self.fy or (0.8 * w)
            self.cx = self.cx or (w / 2.0)
            self.cy = self.cy or (h / 2.0)
            print(f"Using intrinsics fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def get_3d_position_from_depth(self, u, v, depth_map):
        """Compute 3D point (X, Y, Z) in camera coordinates from pixel coords and depth map.

        u, v: float pixel coordinates (x, y)
        depth_map: 2D numpy array with depth in meters
        """
        h, w = depth_map.shape[:2]
        u_i = int(np.clip(round(u), 0, w - 1))
        v_i = int(np.clip(round(v), 0, h - 1))
        z = float(depth_map[v_i, u_i])
        if z <= 0 or np.isnan(z):
            return None
        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def process_frame(self, frame_rgb, depth_map, timestamp=None, write_images=False):
        """Process a single RGB+D frame provided as numpy arrays.

        Returns a tuple (frame_bgr, detected_obj, annotated_image):
            - frame_bgr: original captured BGR image (numpy.ndarray)
            - detected_obj: dict with detection info or None
            - annotated_image: RGB annotated image returned by SAM2 (or None)
        """
        if frame_rgb is None:
            return None, None, None

        self._ensure_intrinsics(frame_rgb)

        if timestamp is None:
            timestamp = time.perf_counter()

        self.frame_count += 1
        current_time = timestamp if self.start_time is None else timestamp - (self.start_time or 0)

        annotated_image = None
        detected_obj = None

        # similar processing logic: either use Kalman predict or detect
        if self.throw_started:
            if self.tracking_active and self.frame_count % 3 == 1 and self.kalman_active and (current_time - (self.last_detected_time or 0) < 0.1):
                filtered_pos, filtered_vel = self.kalman.predict(current_time - (self.last_time or current_time))
                pos_predicted = np.array([filtered_pos[1], filtered_pos[2], filtered_pos[0]])
                vel_predicted = np.array([filtered_vel[1], filtered_vel[2], filtered_vel[0]])
                self.positions_history.append(pos_predicted)
                self.velocities_history.append(vel_predicted)
                self.timestamps_history.append(current_time)
                self.frame_numbers_history.append(self.frame_count)
            else:
                obj, annotated_image = self.detect_thrown_object(frame_rgb, depth_map, current_time)
                if obj:
                    detected_obj = obj
                    bbox = obj["bbox"]
                    cx, cy = obj["centroid"].astype(int)
                    pos_3d = obj["position_3d"]
                    if pos_3d is not None:
                        pos_kalman = np.array([pos_3d[2], pos_3d[0], pos_3d[1]])
                        if not self.kalman_active:
                            if self.last_position is not None and self.last_detected_time is not None:
                                raw_vel = (pos_3d - self.last_position) / (current_time - self.last_detected_time + 1e-6)
                                current_speed = np.linalg.norm(raw_vel)
                                if current_speed > self.motion_threshold:
                                    self.motion_count += 1
                                    self.initial_vel = (raw_vel + self.initial_vel) / 2
                                else:
                                    self.motion_count = 0
                                    self.initial_vel = np.zeros(3)

                                if self.motion_count >= self.min_motion_detections:
                                    vel_kalman = np.array([self.initial_vel[2], self.initial_vel[0], self.initial_vel[1]])
                                    self.kalman.initialize(pos_kalman, vel_kalman, k=0.05)
                                    self.kalman_active = True
                                    self.kalman_active_frame = self.frame_count
                                    filtered_pos = pos_kalman
                                    filtered_vel = vel_kalman
                                else:
                                    filtered_pos = pos_kalman
                                    filtered_vel = np.array([raw_vel[2], raw_vel[0], raw_vel[1]])
                            else:
                                filtered_pos = pos_kalman
                                filtered_vel = np.zeros(3)
                        else:
                            self.kalman.predict(current_time - (self.last_time or current_time))
                            filtered_pos, filtered_vel = self.kalman.update(pos_kalman)

                            # Check if position has been constant
                            if self.last_obj_dict is not None and np.linalg.norm(pos_3d - self.last_obj_dict['position_3d']) < 0.1:
                                self.stagnating_count += 1
                                if self.stagnating_count >= 5:
                                    self.kalman_active = False  # Only disable Kalman, keep motion history
                                    self.stagnating_count = 0
                                    self.motion_count = 0

                        pos_filtered = np.array([filtered_pos[1], filtered_pos[2], filtered_pos[0]])
                        vel_filtered = np.array([filtered_vel[1], filtered_vel[2], filtered_vel[0]])

                        if not self.tracking_active:
                            self.tracking_active = True

                        self.positions_history.append(pos_filtered)
                        self.velocities_history.append(vel_filtered)
                        self.timestamps_history.append(current_time)
                        self.frame_numbers_history.append(self.frame_count)

                        self.last_position = pos_3d
                        self.last_detected_time = current_time
                        self.last_obj_dict = {**obj, "velocity_3d": vel_filtered}

                    # Do not draw on the original frame here. The wrapper will draw the bbox on the displayed frame.
                    if write_images and annotated_image is not None:
                        self.image_buffer.append(annotated_image)
                        self.image_frame_count.append(self.frame_count)
                else:
                    if self.kalman_active and (current_time - (self.last_detected_time or 0) < 0.1):
                        filtered_pos, filtered_vel = self.kalman.predict(current_time - (self.last_time or current_time))
                        pos_predicted = np.array([filtered_pos[1], filtered_pos[2], filtered_pos[0]])
                        vel_predicted = np.array([filtered_vel[1], filtered_vel[2], filtered_vel[0]])
                        self.positions_history.append(pos_predicted)
                        self.velocities_history.append(vel_predicted)
                        self.timestamps_history.append(current_time)
                        self.frame_numbers_history.append(self.frame_count)
        else:
            # Not started; return plain BGR conversion for display and no detection
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr, None, None

        self.last_time = current_time
        # Return original BGR frame (no internal drawing), detected object (or None), and annotated image
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr, detected_obj, annotated_image

    def detect_thrown_object(self, frame, current_depth, current_time, timing=False):
        """Detect using the IncrementalObjectTracker but compute 3D from depth map."""
        if timing:
            t_sam = time.perf_counter()

        if current_depth is None:
            return None, None
        annotated_image = self.tracker.add_image(frame, full_detection=not self.tracking_active)
        if annotated_image is None:
            return None, None
        if timing:
            t_sam_end = time.perf_counter()
            print(f"SAM2 prediction took: {(t_sam_end - t_sam) * 1000:.1f} ms")

        least_displacement = float('inf')
        best_centroid = best_bbox = best_mask = best_position_3d = best_area = None

        for obj_id, obj_info in self.tracker.last_mask_dict.labels.items():
            width_px = max(1, obj_info.x2 - obj_info.x1)
            height_px = max(1, obj_info.y2 - obj_info.y1)

            mask = obj_info.mask
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if xs.numel() == 0:
                centroid = np.array([obj_info.x1 + width_px / 2.0, obj_info.y1 + height_px / 2.0])
            else:
                centroid = np.array([xs.float().mean().item(), ys.float().mean().item()])

            cx = int(np.clip(centroid[0], 0, current_depth.shape[1] - 1))
            cy = int(np.clip(centroid[1], 0, current_depth.shape[0] - 1))
            depth_value = float(current_depth[cy, cx])
            if depth_value <= 0 or np.isnan(depth_value):
                continue
            depth = depth_value

            if depth < self.min_depth:
                continue
            width_m = (width_px * depth) / self.fx
            height_m = (height_px * depth) / self.fy
            area_m2 = width_m * height_m
            if area_m2 < self.min_object_size or area_m2 > self.max_object_size:
                continue

            position_3d = self.get_3d_position_from_depth(centroid[0], centroid[1], current_depth)
            if position_3d is None or np.isnan(position_3d).any():
                continue
            if self.last_position is None or self.last_detected_time is None:
                return {
                    "centroid": centroid,
                    "bbox": (obj_info.x1, obj_info.y1, width_px, height_px),
                    "mask": obj_info.mask,
                    "area": area_m2,
                    "position_3d": position_3d,
                }, annotated_image

            displacement = np.linalg.norm(position_3d - self.last_position)
            time_diff = current_time - self.last_detected_time
            speed = displacement / (time_diff + 1e-6)
            if speed > 10.0:
                continue
            if displacement < least_displacement:
                least_displacement = displacement
                best_mask = obj_info.mask
                best_centroid = centroid
                best_bbox = (obj_info.x1, obj_info.y1, width_px, height_px)
                best_area = area_m2
                best_position_3d = position_3d

        if best_centroid is None:
            return None, annotated_image
        return {
            "centroid": best_centroid,
            "bbox": best_bbox,
            "mask": best_mask,
            "area": best_area,
            "position_3d": best_position_3d,
        }, annotated_image

    def start_recording(self, filename=None):
        if not self.record_video:
            return
        if filename is None:
            filename = f"outputs/videos/array_video_{self.datetime_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.video_fps, self.video_resolution)
        self.recording_started = True

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording_started = False

    def get_curr_pose(self):
        """Get the current estimated position and velocity."""
        if not self.positions_history or not self.velocities_history:
            return None, None
        return np.concatenate((self.positions_history[-1], self.velocities_history[-1]))

    def finalize(self, save_plots=False):
        """Finalize tracking: stop recording, save tracking data, plots and annotated images.

        This centralizes data logging so callers (wrappers / tests) can simply call tracker.finalize().
        """
        if self.recording_started:
            try:
                self.stop_recording()
            except Exception as e:
                print(f"Error stopping recording: {e}")

        # Save tracking data and plots
        try:
            if self.positions_history and self.velocities_history:
                print("Saving tracking data and generating plots...")
                self.data_logger.save_tracking_data(self)
                # Only show/save plots if requested
                try:
                    self.data_logger.plot_tracking_data(self, save_plots=save_plots)
                except Exception as e:
                    print(f"Error generating plots: {e}")
        except Exception as e:
            print(f"Error saving tracking data: {e}")

        # Save annotated images
        try:
            if self.image_buffer:
                print("Saving annotated images...")
                self.data_logger.save_annotated_images(self)
        except Exception as e:
            print(f"Error saving annotated images: {e}")
