import numpy as np
from scipy.linalg import block_diag

class EKF:
    """Extended Kalman Filter for 3D trajectory smoothing with ballistic motion and air resistance.
    
    State vector: [x, y, z, vx, vy, vz, k]
    where k is the air resistance coefficient.
    F_drag = -k * v * |v| acts opposite to velocity.
    """
    
    def __init__(self, 
                 init_pos_std=0.01,    
                 init_vel_std=0.1,      
                 pos_process_std=0.01, 
                 vel_process_std_xy=0.01, 
                 vel_process_std_z=0.5,  
                 meas_std=0.5,
                 k=0.05):        
        """Initialize the Invariant EKF.
        
        Args:
            init_pos_std (float): Initial position uncertainty standard deviation (meters)
            init_vel_std (float): Initial velocity uncertainty standard deviation (m/s)
            pos_process_std (float): Position process noise standard deviation (meters)
            vel_process_std_xy (float): XY velocity process noise standard deviation (m/s)
            vel_process_std_z (float): Z velocity process noise standard deviation (m/s)
            meas_std (float): Measurement noise standard deviation (meters)
            k (float): Air resistance coefficient
        """
        # State dimension (x,y,z, vx,vy,vz, k)
        self.n = 7
        
        # Measurement dimension (x,y,z)
        self.m = 3
        
        # Gravity vector (assuming z is up)
        self.g = np.array([0, 0, -9.81])
        
        # State estimate including air resistance coefficient
        self.x = np.zeros(self.n)
        self.x[6] = 0.05  # Initialize air resistance coefficient
        
        # Initial state covariance (P)
        # Diagonal elements represent initial uncertainty in each state
        self.P = block_diag(
            init_pos_std**2 * np.eye(3),    # position variance
            init_vel_std**2 * np.eye(3),    # velocity variance
            np.array([[0.01**2]])           # air resistance coefficient variance
        )
        
        # Process noise covariance (Q)
        # Represents uncertainty in the motion model
        self.Q = block_diag(
            pos_process_std**2 * np.eye(3),  # position process noise
            np.diag([vel_process_std_xy**2,  # x velocity process noise
                    vel_process_std_xy**2,  # y velocity process noise
                    vel_process_std_z**2]),  # z velocity process noise
            np.array([[0.001**2]])          # air resistance coefficient process noise
        )
        
        # Measurement noise covariance (R)
        # Represents uncertainty in position measurements
        self.R = meas_std**2 * np.eye(self.m)
        
        # Store standard deviations for potential reinitializations
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std
        
        # Initialize system matrices
        # Measurement matrix (directly observes position)
        self.H = np.block([np.eye(3), np.zeros((3, 4))])  # Extended for air resistance state
        
        # First Jacobian calculation
        self._calculate_jacobian(1/30)
    
    def _calculate_jacobian(self, dt):
        """Update the Jacobian of the state transition function (F matrix).
        
        State transition with air resistance:
        x[t+1] = x[t] + vx[t]*dt
        y[t+1] = y[t] + vy[t]*dt
        z[t+1] = z[t] + vz[t]*dt + 0.5*g*dt^2
        vx[t+1] = vx[t] - k*vx[t]*|v|*dt
        vy[t+1] = vy[t] - k*vy[t]*|v|*dt
        vz[t+1] = vz[t] + g*dt - k*vz[t]*|v|*dt
        k[t+1] = k[t]
        
        where |v| = sqrt(vx^2 + vy^2 + vz^2)
        """
        vel = self.x[3:6]
        k = self.x[6]
        v_mag = np.linalg.norm(vel)
        
        # Partial derivatives for velocity update
        if v_mag > 1e-6:  # Avoid division by zero
            dv_dv = np.eye(3) - k * dt * (
                np.outer(vel, vel) / v_mag + v_mag * np.eye(3)
            )
        else:
            dv_dv = np.eye(3)
        
        # Partial derivatives with respect to k
        dv_dk = -dt * v_mag * vel
        
        # Complete Jacobian matrix
        self.F = np.zeros((7, 7))
        self.F[0:3, 0:3] = np.eye(3)          # d(pos)/d(pos)
        self.F[0:3, 3:6] = dt * np.eye(3)     # d(pos)/d(vel)
        self.F[3:6, 3:6] = dv_dv              # d(vel)/d(vel)
        self.F[3:6, 6] = dv_dk                # d(vel)/d(k)
        self.F[6, 6] = 1.0                    # d(k)/d(k)

    def predict(self, dt):
        """Predict step of the Kalman filter using ballistic motion with air resistance."""
        # Extract current state
        pos = self.x[:3]
        vel = self.x[3:6]
        k = self.x[6]
        
        # Calculate air resistance force
        v_mag = np.linalg.norm(vel)
        if v_mag > 1e-6:
            f_drag = -k * vel * v_mag  # F = -k*v*|v|
        else:
            f_drag = np.zeros(3)
        
        # Predict state using motion equations with air resistance
        # Position: p(t) = p0 + v0*t + 0.5*(g + f_drag/m)*t^2
        # Velocity: v(t) = v0 + (g + f_drag/m)*t
        new_pos = pos + vel * dt + 0.5 * (self.g + f_drag) * dt**2
        new_vel = vel + (self.g + f_drag) * dt

        # Update state (air resistance coefficient stays constant)
        self.x = np.concatenate([new_pos, new_vel, [self.x[6]]])
        
        # Update the Jacobian for the current state
        self._calculate_jacobian(dt)
        
        # Predict covariance using standard EKF form
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:3], self.x[3:6]  # return position and velocity
    
    def update(self, z):
        """Update step of the Kalman filter.
        
        Args:
            z (numpy.ndarray): Measurement vector [x, y, z]
        """
        if z is None or np.any(np.isnan(z)):
            return self.x[:3], self.x[3:]
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance using Joseph form for numerical stability
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x[:3], self.x[3:]  # return position and velocity
    


    def initialize(self, pos, vel=None, k=0.05):
        """Initialize the filter with a position and optionally velocity and air resistance.
        
        Args:
            pos (numpy.ndarray): Initial position [x, y, z]
            vel (numpy.ndarray, optional): Initial velocity [vx, vy, vz]
            k (float, optional): Initial air resistance coefficient
        """
        # Set initial state
        self.x[:3] = pos
        if vel is not None:
            self.x[3:6] = vel
            vel_std = self.init_vel_std
        else:
            self.x[3:6] = np.zeros(3)
            vel_std = 1.0  # Higher uncertainty if velocity unknown
        
        self.x[6] = k
        
        # Set initial covariance using stored standard deviations
        self.P = block_diag(
            self.init_pos_std**2 * np.eye(3),    # position variance
            vel_std**2 * np.eye(3),              # velocity variance
            np.array([[0.01**2]])                # air resistance variance
        )