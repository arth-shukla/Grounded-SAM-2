import pyzed.sl as sl
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from collections import deque
import time
import os
import csv
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker

COLOR_DIST_THRESHOLD = 0.03


class ObjectThrowTracker:
    def __init__(
        self,
        prompt_text,
        detection_interval = 4,
        max_history=300,
        gaussian_sigma=2.0,
        min_speed_threshold=0.5,
        max_object_size=0.05, # in square meters, overestimate
        min_object_size=0.0,
        min_depth=0.3,
        record_video=True,
        video_fps=60,
        video_resolution=(640, 360),
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
        track_rotation=False
    ):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
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
            sam2_ckpt_path=sam2_checkpoint,
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

        self.last_position = None
        self.last_time = None
        self.last_obj_dict = None
        self.start_time = time.time()
        self.frame_count = 0
        self.tracking_active = False
        self.throw_started = False

        self.background_frame = None
        self.background_gray = None
        self.background_depth = None

        self.processing_times = deque(maxlen=100)
        
        # Video recording settings
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_resolution = video_resolution
        self.video_writer = None
        self.recording_started = False

        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("Initialized!")
        
        self.capture_background()
        print("\n" + "=" * 60)
        print("READY TO TRACK!")
        print(f"Video Recording: {'ENABLED' if self.record_video else 'DISABLED'}")
        print("=" * 60 + "\n")

    def capture_background(self):
        """Capture and store the background frame and depth."""
        print("Capturing background...")
        for _ in range(30):
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.DEPTH)
                
        # Get the RGB frame
        self.background_frame = self.image.get_data().copy()
        self.background_gray = cv2.cvtColor(self.background_frame, cv2.COLOR_BGRA2GRAY)
        
        # Get the depth data
        depth_mat = sl.Mat()
        self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        self.background_depth = depth_mat.get_data().copy()
        self.background_depth_mask = (self.background_depth > 0) & (self.background_depth < 10.0)
        
        print("Background captured successfully.")
        # Print depth statistics to verify correct capture
        valid_depth = self.background_depth[self.background_depth_mask]
        if valid_depth.size > 0:
            print(f"Depth stats - Min: {valid_depth.min():.3f}m, Max: {valid_depth.max():.3f}m, Mean: {valid_depth.mean():.3f}m")
        else:
            print("Warning: No valid depth data in background capture")

    def get_3d_position(self, x, y):
        """Get XYZ coordinates from depth map."""
        x = round(x)
        y = round(y)
        err, point = self.point_cloud.get_value(x, y)
        if err == sl.ERROR_CODE.SUCCESS:
            return np.array([point[0], point[1], point[2]])
        return None

    def calculate_velocity(self, position, current_time):
        """Estimate velocity from position delta over time."""
        if self.last_position is None or self.last_time is None:
            return np.array([0.0, 0.0, 0.0])
        dt = current_time - self.last_time
        if dt == 0:
            return np.array([0.0, 0.0, 0.0])
        return (position - self.last_position) / dt

    def track_frame(self):
        """Process a single frame and track the object."""
        frame_start = time.perf_counter()

        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.RIGHT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)

            depth_mat = sl.Mat()
            self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            current_depth = depth_mat.get_data()

            frame = self.image.get_data() # BGRA format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Store current frame for mouse callback
            self.current_frame = frame_bgr.copy()
            tracked = False

            if self.throw_started:
                current_time = time.time() - self.start_time
                self.frame_count += 1
                print('Frame:', self.frame_count, 'Camera fps:', self.zed.get_current_fps())
                
                time1 = time.perf_counter()
                obj, annotated_image = self.detect_thrown_object(frame_rgb, current_depth, timing=True)
                print(f"detect_thrown_object used: {(time.perf_counter() - time1) * 1000:.3f} ms")
                if obj:
                    bbox = obj["bbox"]
                    cx, cy = obj["centroid"].astype(int)
                    pos_3d = self.get_3d_position(cx, cy)
                    if pos_3d is not None:
                        vel_3d = self.calculate_velocity(pos_3d, current_time)

                        if not self.tracking_active:
                            self.tracking_active = True
                        
                        self.positions_history.append(pos_3d)
                        self.velocities_history.append(vel_3d)
                        self.timestamps_history.append(current_time)
                        self.frame_numbers_history.append(self.frame_count)

                        self.last_position = pos_3d
                        self.last_time = current_time
                        self.last_obj_dict = obj
                        tracked = True

                        x, y, w, h = bbox
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)

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

            return tracked, frame_bgr

        return False, None
    
    def detect_thrown_object(self, frame, current_depth=None, timing=False):
        """Track object using Grounded SAM2 predictor.
        
        Args:
            frame: Input frame to process
            current_depth: Depth map corresponding to the frame
            timing: If True, print timing information for each step
        """
        if timing:
            t_sam = time.perf_counter()
            
        annotated_image = self.tracker.add_image(frame)
        
        if timing:
            t_sam_end = time.perf_counter()
            print(f"SAM2 prediction took: {(t_sam_end - t_sam) * 1000:.1f} ms")
            
        if annotated_image is not None and isinstance(annotated_image, np.ndarray):
            out_dir =  os.path.join("./outputs", self.datetime_str, "result")
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"annotated_frame{self.frame_count}.png")
            cv2.imwrite(filename, annotated_image)

        least_displacement = float('inf')
        best_centroid, best_bbox, best_mask = None, None, None
        print('Number of tracked objects:', len(self.tracker.last_mask_dict.labels))
        for obj_id, obj_info in self.tracker.last_mask_dict.labels.items():
            width_px = max(1, obj_info.x2 - obj_info.x1)
            height_px = max(1, obj_info.y2 - obj_info.y1)

            centroid = np.array([obj_info.x1 + obj_info.x2, obj_info.y1 + obj_info.y2]) / 2

            depth = None
            if current_depth is not None:
                cx = int(np.clip(centroid[0], 0, current_depth.shape[1] - 1))
                cy = int(np.clip(centroid[1], 0, current_depth.shape[0] - 1))
                depth_value = float(current_depth[cy, cx])
                if depth_value > 0:
                    depth = depth_value
                else:
                    print(f"Warning: Invalid depth value {depth_value} at centroid for object {obj_id}.")
            else:
                print("Warning: Current depth not found.")
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
            if self.last_position is None: # Just pick the first valid detection
                return {
                    "centroid": centroid,
                    "bbox": (obj_info.x1, obj_info.y1, width_px, height_px),
                    "mask": obj_info.mask,
                    "area": area_m2
                }, annotated_image
            displacement = np.linalg.norm(position_3d - self.last_position)
            if displacement < least_displacement:
                least_displacement = displacement
                best_mask = obj_info.mask
                best_centroid = centroid
                best_bbox = (obj_info.x1, obj_info.y1, width_px, height_px)
                best_area = area_m2 if depth is not None else None

        if best_centroid is None:
            print('Warning: Object not detected by Grounded SAM2')
            return None, None
        print('best_area:', best_area)
        return {
            "centroid": best_centroid,
            "bbox": best_bbox,
            "mask": best_mask,
            "area": best_area
        }, annotated_image

    def run(self):
        """Main loop."""
        print("\nControls:")
        print("  SPACE       - Start tracking mode")
        print("  S           - Start/Stop recording")
        # print("  R           - Reset background")
        print("  P           - Generate plots")
        print("  Q           - Quit")
        
        while True:
            if self.record_video and self.throw_started and not self.recording_started:
                self.start_recording()
            time1 = time.perf_counter()
            tracked, frame = self.track_frame()
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
                if self.recording_started:
                    self.stop_recording()
                
                if self.positions_history and self.velocities_history:
                    print("Saving tracking data...")
                    self.save_tracking_data()
                    print("Generating plots...")
                    self.plot_tracking_data(save_plots=False)
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
                    self.plot_tracking_data(save_plots=True)
                else:
                    print("\nNo tracking data available to plot yet.")
            # elif key == ord("r"):
            #     print("\nRecapturing background...")
            #     self.capture_background()
            #     self.reset_tracking()
            #     self.throw_started = False
            #     self.tracking_active = False

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
            
    def save_tracking_data(self):
        """Save position and velocity data to CSV file."""
        if not self.positions_history or not self.velocities_history:
            print("\nNo tracking data to save.")
            return False
            
        timestamp =  self.datetime_str
        filename = f"tracking_data.csv"
        filepath = os.path.join('outputs', self.datetime_str, 'tracking_data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header based on whether rotation tracking is enabled
                header = [
                    'frame_number', 'timestamp', 
                    'pos_x', 'pos_y', 'pos_z',
                    'vel_x', 'vel_y', 'vel_z', 'fps'
                ]
                
                writer.writerow(header)
                
                for i in range(len(self.timestamps_history)):
                    if i < len(self.positions_history) and i < len(self.velocities_history):
                        pos = self.positions_history[i]
                        vel = self.velocities_history[i]
                        # fps = self.fps_history[i] if i < len(self.fps_history) else -1
                        timestamp = self.timestamps_history[i]
                        frame = self.frame_numbers_history[i] if i < len(self.frame_numbers_history) else 0
                        row = [
                            frame, timestamp,
                            pos[0], pos[1], pos[2],
                            vel[0], vel[1], vel[2], self.zed.get_current_fps()
                        ]
                        writer.writerow(row)
                
            print(f"\n✓ Tracking data saved to {filename}")
            print(f"  - {len(self.positions_history)} position records")
            print(f"  - {len(self.velocities_history)} velocity records")
            if self.track_rotation and self.orientations_history:
                print(f"  - {len(self.orientations_history)} orientation records")
            return True
            
        except Exception as e:
            print(f"\nError saving tracking data: {e}")
            return False

    def plot_tracking_data(self, save_plots=True):
        """Generate comprehensive plots of tracking data."""
        if not self.positions_history or not self.velocities_history:
            print("\nNo tracking data to plot.")
            return
        
        print("\n📊 Generating plots...")
        
        # Convert lists to numpy arrays for easier manipulation
        positions = np.array(self.positions_history)
        velocities = np.array(self.velocities_history)
        timestamps = np.array(self.timestamps_history)
        
        # Calculate speed (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Create timestamp for plot filenames
        timestamp_str = self.datetime_str
        
        # Determine subplot layout based on rotation tracking
        if self.track_rotation and self.orientations_history:
            # 3x3 layout for rotation data
            fig = plt.figure(figsize=(20, 15))
            subplot_layout = (3, 3)
            
            # Convert orientation data to arrays (convert radians to degrees for plotting)
            orientations = np.array([np.degrees(o['angle']) for o in self.orientations_history])
            angular_velocities = np.array(self.angular_velocities_history)
            aspect_ratios = np.array([o['aspect_ratio'] for o in self.orientations_history])
        else:
            # 2x3 layout for standard data
            fig = plt.figure(figsize=(16, 12))
            subplot_layout = (2, 3)
        
        # 1. 3D Trajectory Plot
        ax1 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 1, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, alpha=0.6)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   c='red', s=100, marker='X', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Set identical scales for x, y, z axes
        ax1.set_xlim(-3.0, 3.0)
        ax1.set_ylim(-3.0, 3.0)
        ax1.set_zlim(-3.0, 3.0)
        # Make the plot aspect ratio equal
        ax1.set_box_aspect([1, 1, 1])
        
        # 2. Position vs Time (X, Y, Z)
        ax2 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 2)
        ax2.plot(timestamps, positions[:, 0], 'r-', label='X', linewidth=2)
        ax2.plot(timestamps, positions[:, 1], 'g-', label='Y', linewidth=2)
        ax2.plot(timestamps, positions[:, 2], 'b-', label='Z', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocity vs Time (Vx, Vy, Vz)
        ax3 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 3)
        ax3.plot(timestamps, velocities[:, 0], 'r-', label='Vx', linewidth=2)
        ax3.plot(timestamps, velocities[:, 1], 'g-', label='Vy', linewidth=2)
        ax3.plot(timestamps, velocities[:, 2], 'b-', label='Vz', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity Components vs Time', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Z vs Y Position
        ax4 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 4)
        ax4.plot(positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax4.scatter(positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
        ax4.scatter(positions[-1, 1], positions[-1, 2], c='red', s=100, marker='X', label='End')
        ax4.set_xlabel('Y Position (m)')
        ax4.set_ylabel('Z Position (m)')
        ax4.set_title('Z vs Y Position', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        ax4.set_xlim(-3.0, 3.0)
        ax4.set_ylim(-3.0, 3.0)
        ax4.legend()

        # 5. Speed vs Time
        ax5 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 5)
        ax5.plot(timestamps, speeds, 'purple', linewidth=2)
        ax5.fill_between(timestamps, speeds, alpha=0.3, color='purple')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Speed (m/s)')
        ax5.set_title('Speed vs Time', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add statistics
        max_speed = np.max(speeds)
        avg_speed = np.mean(speeds)
        ax5.axhline(y=avg_speed, color='orange', linestyle='--', 
                   label=f'Avg: {avg_speed:.2f} m/s')
        ax5.axhline(y=max_speed, color='red', linestyle='--', 
                   label=f'Max: {max_speed:.2f} m/s')
        ax5.legend()
        
        # Calculate statistics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        duration = timestamps[-1] - timestamps[0]
        displacement = np.linalg.norm(positions[-1] - positions[0])

        # FPS / processing time statistics (self.processing_times stores ms per frame)
        fps_stats_text = "N/A"
        if hasattr(self, 'processing_times') and len(self.processing_times) > 0:
            proc_times = np.array(self.processing_times)
            mean_ms = np.mean(proc_times)
            median_ms = np.median(proc_times)
            min_ms = np.min(proc_times)
            max_ms = np.max(proc_times)
            mean_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            median_fps = 1000.0 / median_ms if median_ms > 0 else 0.0
            min_fps = 1000.0 / max_ms if max_ms > 0 else 0.0
            max_fps = 1000.0 / min_ms if min_ms > 0 else 0.0
            fps_stats_text = (
                f"FPS (mean/med/min/max): {mean_fps:.1f}/{median_fps:.1f}/{min_fps:.1f}/{max_fps:.1f}"
                f"  | proc_ms (mean/med/min/max): {mean_ms:.1f}/{median_ms:.1f}/{min_ms:.1f}/{max_ms:.1f}"
            )

        stats_text = f"""
        TRACKING STATISTICS
        ═══════════════════════════════
        
        Duration: {duration:.3f} s
        Data Points: {len(positions)}
        
        POSITION
        Start: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f}) m
        End: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}) m
        Displacement: {displacement:.3f} m
        Total Distance: {total_distance:.3f} m
        
        VELOCITY
        Max Speed: {max_speed:.3f} m/s
        Avg Speed: {avg_speed:.3f} m/s
        Min Speed: {np.min(speeds):.3f} m/s
        
        Max Vx: {np.max(velocities[:, 0]):.3f} m/s
        Max Vy: {np.max(velocities[:, 1]):.3f} m/s
        Max Vz: {np.max(velocities[:, 2]):.3f} m/s"""
        
        # Append FPS stats to stats_text
        stats_text += f"\n\n{fps_stats_text}\n"

        plt.tight_layout()
        
        if save_plots:
            plot_filename = f"tracking_plots_{timestamp_str}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to {plot_filename}")
        
        plt.show()
        
        print("✓ Plot generation complete!")


if __name__ == "__main__":
    tracker = ObjectThrowTracker(
        prompt_text="banana.",
        # max_object_size=0.3,  # Adjusted for larger objects
        detection_interval=10,
        max_history=300,
        record_video=True,
        video_resolution=(1280, 720),
        track_rotation=False
    )
    tracker.run()
