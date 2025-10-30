
import argparse
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TrackingData:
    """Class for holding tracking data read from CSV files."""
    frame_numbers: np.ndarray
    timestamps: np.ndarray
    positions: np.ndarray  # shape: (N, 3) for x, y, z
    velocities: np.ndarray  # shape: (N, 3) for vx, vy, vz
    fps: np.ndarray

class TrackingDataLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def save_tracking_data(self, object_tracker):
        """Save position and velocity data to CSV file."""
        if not object_tracker.positions_history or not object_tracker.velocities_history:
            print("\nNo tracking data to save.")
            return False
            
        timestamp =  object_tracker.datetime_str
        filename = f"tracking_data.csv"
        filepath = os.path.join(self.log_dir, object_tracker.datetime_str, 'tracking_data', filename)
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
                
                for i in range(len(object_tracker.timestamps_history)):
                    if i < len(object_tracker.positions_history) and i < len(object_tracker.velocities_history):
                        pos = object_tracker.positions_history[i]
                        vel = object_tracker.velocities_history[i]
                        # fps = object_tracker.fps_history[i] if i < len(object_tracker.fps_history) else -1
                        timestamp = object_tracker.timestamps_history[i]
                        frame = object_tracker.frame_numbers_history[i] if i < len(object_tracker.frame_numbers_history) else 0
                        row = [
                            frame, timestamp,
                            pos[0], pos[1], pos[2],
                            vel[0], vel[1], vel[2], object_tracker.fps_history[i] if i < len(object_tracker.fps_history) else -1
                        ]
                        writer.writerow(row)
                
            print(f"\n✓ Tracking data saved to {filename}")
            print(f"  - {len(object_tracker.positions_history)} position records")
            print(f"  - {len(object_tracker.velocities_history)} velocity records")
            if object_tracker.track_rotation and object_tracker.orientations_history:
                print(f"  - {len(object_tracker.orientations_history)} orientation records")
            return True
            
        except Exception as e:
            print(f"\nError saving tracking data: {e}")
            return False

    def plot_tracking_data(self, object_tracker, save_plots=True):
        """Generate comprehensive plots of tracking data."""
        if not object_tracker.positions_history or not object_tracker.velocities_history:
            print("\nNo tracking data to plot.")
            return
        
        print("\n📊 Generating plots...")
        
        # Convert lists to numpy arrays for easier manipulation
        positions = np.array(object_tracker.positions_history)
        velocities = np.array(object_tracker.velocities_history)
        timestamps = np.array(object_tracker.timestamps_history)
        
        # Calculate speed (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Create timestamp for plot filenames
        timestamp_str = object_tracker.datetime_str
        
        # Determine subplot layout based on rotation tracking
        if object_tracker.track_rotation and object_tracker.orientations_history:
            # 3x3 layout for rotation data
            fig = plt.figure(figsize=(20, 15))
            subplot_layout = (3, 3)
            
            # Convert orientation data to arrays (convert radians to degrees for plotting)
            orientations = np.array([np.degrees(o['angle']) for o in object_tracker.orientations_history])
            angular_velocities = np.array(object_tracker.angular_velocities_history)
            aspect_ratios = np.array([o['aspect_ratio'] for o in object_tracker.orientations_history])
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
        
        # 4. Y vs. X Position
        ax4 = fig.add_subplot(subplot_layout[0], subplot_layout[1], 4)
        ax4.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax4.scatter(positions[0, 1], positions[0, 0], c='green', s=100, marker='o', label='Start')
        ax4.scatter(positions[-1, 1], positions[-1, 0], c='red', s=100, marker='X', label='End')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y vs X Position', fontsize=12, fontweight='bold')
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

        # FPS / processing time statistics (object_tracker.processing_times stores ms per frame)
        fps_stats_text = "N/A"
        if hasattr(object_tracker, 'processing_times') and len(object_tracker.processing_times) > 0:
            proc_times = np.array(object_tracker.processing_times)
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
            plot_filename = os.path.join(self.log_dir, object_tracker.datetime_str, 'tracking_plots', plot_filename)
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to {plot_filename}")
        
        plt.show()
        
        print("✓ Plot generation complete!")

    def save_annotated_images(self, object_tracker):
        """Save annotated images from buffer to disk."""
        if not object_tracker.image_buffer:
            print("\nNo annotated images to save.")
            return
        
        print("\nSaving annotated images...")
        
        out_dir = os.path.join(self.log_dir, object_tracker.datetime_str, "annotated_images")
        os.makedirs(out_dir, exist_ok=True)
        
        for idx, image in enumerate(object_tracker.image_buffer):
            frame_count = object_tracker.image_frame_count[idx] if idx < len(object_tracker.image_frame_count) else idx
            filename = os.path.join(out_dir, f"annotated_frame{frame_count}.png")
            cv2.imwrite(filename, image)
    def read_tracking_data(filepath: str) -> TrackingData:
        """Read tracking data from a CSV file.
        
        Args:
            filepath: Path to the tracking data CSV file
            
        Returns:
            TrackingData object containing the parsed data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tracking data file not found: {filepath}")
            
        try:
            # Initialize lists to store data
            frame_numbers = []
            timestamps = []
            positions = []
            velocities = []
            fps_values = []
            
            with open(filepath, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Verify required columns exist
                required_columns = {
                    'frame_number', 'timestamp',
                    'pos_x', 'pos_y', 'pos_z',
                    'vel_x', 'vel_y', 'vel_z', 'fps'
                }
                if not required_columns.issubset(set(reader.fieldnames)):
                    missing = required_columns - set(reader.fieldnames)
                    raise ValueError(f"Missing required columns: {missing}")
                
                for row in reader:
                    # Parse each row
                    frame_numbers.append(int(float(row['frame_number'])))
                    timestamps.append(float(row['timestamp']))
                    
                    # Parse positions
                    pos = np.array([
                        float(row['pos_x']),
                        float(row['pos_y']),
                        float(row['pos_z'])
                    ])
                    positions.append(pos)
                    
                    # Parse velocities
                    vel = np.array([
                        float(row['vel_x']),
                        float(row['vel_y']),
                        float(row['vel_z'])
                    ])
                    velocities.append(vel)
                    
                    # Parse FPS
                    fps_values.append(float(row['fps']))
            
            # Convert lists to numpy arrays
            return TrackingData(
                frame_numbers=np.array(frame_numbers),
                timestamps=np.array(timestamps),
                positions=np.array(positions),
                velocities=np.array(velocities),
                fps=np.array(fps_values)
            )
            
        except Exception as e:
            raise ValueError(f"Error reading tracking data: {str(e)}")

    @staticmethod
    def analyze_and_plot_data(data: TrackingData, save_path: str = None):
        """Generate comprehensive plots and analysis from tracking data.
        
        Args:
            data: TrackingData object containing the parsed data
            save_path: Optional path to save the plot. If None, plot is only displayed.
        """
        print("\n📊 Generating plots...")
        
        # Calculate derived quantities
        speeds = np.linalg.norm(data.velocities, axis=1)
        total_distance = np.sum(np.linalg.norm(np.diff(data.positions, axis=0), axis=1))
        duration = data.timestamps[-1] - data.timestamps[0]
        displacement = np.linalg.norm(data.positions[-1] - data.positions[0])
        
        # Create plots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D Trajectory Plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(data.positions[:, 0], data.positions[:, 1], data.positions[:, 2], 
                'b-', linewidth=2, alpha=0.6)
        ax1.scatter(data.positions[0, 0], data.positions[0, 1], data.positions[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(data.positions[-1, 0], data.positions[-1, 1], data.positions[-1, 2], 
                   c='red', s=100, marker='X', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3.0, 3.0)
        ax1.set_ylim(-3.0, 3.0)
        ax1.set_zlim(-3.0, 3.0)
        ax1.set_box_aspect([1, 1, 1])
        
        # 2. Position vs Time
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(data.timestamps, data.positions[:, 0], 'r-', label='X', linewidth=2)
        ax2.plot(data.timestamps, data.positions[:, 1], 'g-', label='Y', linewidth=2)
        ax2.plot(data.timestamps, data.positions[:, 2], 'b-', label='Z', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocity vs Time
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(data.timestamps, data.velocities[:, 0], 'r-', label='Vx', linewidth=2)
        ax3.plot(data.timestamps, data.velocities[:, 1], 'g-', label='Vy', linewidth=2)
        ax3.plot(data.timestamps, data.velocities[:, 2], 'b-', label='Vz', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity Components vs Time', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Y vs X Position
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(data.positions[:, 1], data.positions[:, 0], 'b-', linewidth=2)
        ax4.scatter(data.positions[0, 1], data.positions[0, 0], c='green', s=100, marker='o', label='Start')
        ax4.scatter(data.positions[-1, 1], data.positions[-1, 0], c='red', s=100, marker='X', label='End')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y vs X Position', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        ax4.set_xlim(-3.0, 3.0)
        ax4.set_ylim(-3.0, 3.0)
        ax4.legend()

        # 5. Speed vs Time
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(data.timestamps, speeds, 'purple', linewidth=2)
        ax5.fill_between(data.timestamps, speeds, alpha=0.3, color='purple')
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

        # Print statistics
        stats_text = f"""
        TRACKING STATISTICS
        ═══════════════════════════════
        
        Duration: {duration:.3f} s
        Data Points: {len(data.positions)}
        
        POSITION
        Start: ({data.positions[0, 0]:.3f}, {data.positions[0, 1]:.3f}, {data.positions[0, 2]:.3f}) m
        End: ({data.positions[-1, 0]:.3f}, {data.positions[-1, 1]:.3f}, {data.positions[-1, 2]:.3f}) m
        Displacement: {displacement:.3f} m
        Total Distance: {total_distance:.3f} m
        
        VELOCITY
        Max Speed: {max_speed:.3f} m/s
        Avg Speed: {avg_speed:.3f} m/s
        Min Speed: {np.min(speeds):.3f} m/s
        
        Max Vx: {np.max(data.velocities[:, 0]):.3f} m/s
        Max Vy: {np.max(data.velocities[:, 1]):.3f} m/s
        Max Vz: {np.max(data.velocities[:, 2]):.3f} m/s
        
        FPS Statistics
        Mean FPS: {np.mean(data.fps):.1f}
        Min FPS: {np.min(data.fps):.1f}
        Max FPS: {np.max(data.fps):.1f}
        """
        print(stats_text)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to {save_path}")
        
        plt.show()
        print("✓ Plot generation complete!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tracking data from CSV file")
    parser.add_argument("filepath", type=str, help="Path to the tracking data CSV file")
    parser.add_argument("--save", type=str, help="Path to save the plot (optional)", default=None)
    args = parser.parse_args()

    try:
        # Read the data
        data = TrackingDataLogger.read_tracking_data(args.filepath)
        
        # Create plots and print statistics
        TrackingDataLogger.analyze_and_plot_data(data, args.save)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)