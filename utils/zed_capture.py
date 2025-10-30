import pyzed.sl as sl
import numpy as np
import cv2
import os
import argparse
from datetime import datetime

def capture_zed_video(output_dir, duration=None):
    """
    Capture video from ZED camera, saving both RGB and depth data.
    
    Args:
        output_dir (str): Directory to save the output files
        duration (float, optional): Duration in seconds to record. If None, records until 'q' is pressed.
    """
    # Initialize ZED camera
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p resolution
    init_params.camera_fps = 60 
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode for better accuracy
    init_params.coordinate_units = sl.UNIT.METER  # Set depth units to meters
    init_params.depth_minimum_distance = 0.3
    init_params.depth_maximum_distance = 10.0

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        return

    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join(output_dir, f"zed_capture_{timestamp}")
    os.makedirs(video_dir, exist_ok=True)

    # Initialize file paths
    rgb_path = os.path.join(video_dir, "rgb.avi")
    xyz_path = os.path.join(video_dir, "xyz.npy")  # Store XYZ point cloud data
    depth_path = os.path.join(video_dir, "depth.npy")  # Store depth data
    
    # Get camera resolution
    camera_config = zed.get_camera_information().camera_configuration
    width = camera_config.resolution.width
    height = camera_config.resolution.height
    
    print(f"Camera resolution: {width}x{height}")

    # Initialize video writer for RGB
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 60, (width, height))
    
    # Initialize list to store point cloud frames
    xyz_frames = []
    depth_frames = []

    start_time = datetime.now()
    frame_count = 0

    zed_runtime = sl.RuntimeParameters()

    print("Recording started. Press 'q' to stop...")

    try:
        while True:
            # Check if the duration has elapsed
            if duration is not None:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time >= duration:
                    break

            # Grab a new frame from the ZED
            if zed.grab(zed_runtime) == sl.ERROR_CODE.SUCCESS:
                # Retrieve RGB image
                image = sl.Mat()
                point_cloud = sl.Mat()
                depth = sl.Mat()
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_data = image.get_data()
                # Convert from RGB (ZED) to BGR (OpenCV)
                rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)

                # Retrieve XYZ point cloud
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                xyz_data = point_cloud.get_data()

                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_data = depth.get_data()
                
                # Log point cloud stats
                if np.isnan(xyz_data).all():
                    print('Point cloud has no valid values (all NaN)')
                else:
                    valid_xyz = xyz_data[~np.isnan(xyz_data).any(axis=2)]
                    if len(valid_xyz) > 0:
                        print('XYZ range:')
                        print(f'  X: {np.min(valid_xyz[:,0]):.2f} to {np.max(valid_xyz[:,0]):.2f} m')
                        print(f'  Y: {np.min(valid_xyz[:,1]):.2f} to {np.max(valid_xyz[:,1]):.2f} m')
                        print(f'  Z: {np.min(valid_xyz[:,2]):.2f} to {np.max(valid_xyz[:,2]):.2f} m')
                print('Point cloud nan %:', np.isnan(xyz_data).any(axis=2).sum() / xyz_data.shape[0] / xyz_data.shape[1] * 100)

                # Write RGB frame
                rgb_writer.write(rgb_frame)
                
                # Store point cloud frame
                xyz_frames.append(xyz_data.copy())
                depth_frames.append(depth_data.copy())

                frame_count += 1

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Clean up
        rgb_writer.release()
        zed.close()
        cv2.destroyAllWindows()

        # Save all point cloud frames as a single .npy file
        xyz_frames = np.array(xyz_frames)  # Convert to numpy array (frames, height, width, xyz)
        np.save(xyz_path, xyz_frames)
        depth_frames = np.array(depth_frames)  # Convert to numpy array (frames, height, width)
        np.save(depth_path, depth_frames)

        print(f"\nRecording finished. Saved {frame_count} frames.")
        print(f"RGB video saved to: {rgb_path}")
        print(f"Point cloud data saved to: {xyz_path}")
        print(f"Point cloud data shape: {xyz_frames.shape} (frames × height × width × xyz)")
        print(f"XYZ units: meters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture RGB and depth video from ZED camera")
    parser.add_argument("--output", type=str, default="throw_recordings", 
                      help="Output directory for saving videos")
    parser.add_argument("--duration", type=float, default=None,
                      help="Duration in seconds to record (optional)")
    
    args = parser.parse_args()
    
    capture_zed_video(args.output, args.duration)