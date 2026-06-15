import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

def ax_fine_grid(ax: Axes) -> None:
    '''Fine grid which allows you to pass an axis.'''
    ax.grid(which='major', color='lightgray', linestyle='-', linewidth=0.7, zorder = 0)
    ax.minorticks_on()
    ax.grid(which='minor',color='lightgray',linestyle='--', linewidth=0.5, zorder = 0)

def trim_after_gps(df: pd.DataFrame, sys_time_header: str = 'system_time_ns', threshold: float = 5e9, jump_to_use: int = 0):

    # Trimming Dataframe to where GPS was first received
    df_copy = df.copy()
    df_copy['diff'] = df[sys_time_header].diff()
    sudden_jump_indices = df_copy.index[df_copy['diff'] > threshold].tolist()

    if len(sudden_jump_indices) > 0:
        # print(f"GPS Time Jump Indices: {sudden_jump_indices}")
        gps_lock_idx = sudden_jump_indices[jump_to_use]
        df_copy = df_copy[gps_lock_idx:]    # trim the data to after gps lock
    else:
        print(f"No GPS Time jumps found in dataframe. Nothing to trim.")
    
    return df_copy, sudden_jump_indices

class TrajectoryReplayer():
    '''
    Creates a 2D or 3D plot that has an associated time slider which 
    allows for replay of a trajectory.

    Requires:
        1. time series
        2. North position series
        3. East position series
        4. Down position series (optional)

    '''
    def __init__(self,
                 time: np.ndarray,
                 north_pos: np.ndarray,
                 east_pos: np.ndarray,
                 down_pos: np.ndarray = None,
                 time_units = 's',
                 position_units = 'm'):
        
        # Check that arrays lengths are the same
        time_len = len(time)
        north_len = len(north_pos)
        east_len = len(east_pos)
        if time_len != north_len:
            raise Exception(f"Mismatching dimensions: Time Array Length {time_len} North Array Length: {north_len}")
        if time_len != east_len:
            raise Exception(f"Mismatching dimensions: Time Array Length {time_len} East Array Length: {east_len}")
        if (down_pos is not None) and (time_len != len(down_pos)):
            raise Exception(f"Mismatching dimensions: Time Array Length {time_len} Down Array Length: {len(down_pos)}")

        self._times = time
        self._north_positions = north_pos
        self._east_positions = east_pos        

        if down_pos is not None:
            self._down_positions = down_pos
            self._down_available = True
        else:
            self._down_available = False

        self._time_units = time_units
        self._position_units = position_units
        self._num_indices = time_len

    def set_time_units(self, units: str):
        self._time_units = units

    def set_position_units(self, units: str):
        self._position_units = units

    def _make_time_slider(self):
        # Time Slider
        ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
        first_time, last_time = self._times[0], self._times[-1]
        step_size = (last_time - first_time) / self._num_indices
        return Slider(ax_slider, f"Time ({self._time_units})", first_time, last_time, valinit = first_time, valstep = step_size)

    def planar_trajectory(self, n_lim: list = None, e_lim: list = None, title_override: str = None, poi: list = None):

        fig, ax = plt.subplots(figsize=(10,6))
        if title_override is not None:
            ax.set_title(title_override)
        else:
            ax.set_title('Planar (NE) Trajectory')
        ax.set_xlabel(f'East ({self._position_units})')
        ax.set_ylabel(f'North ({self._position_units})')
        ax.set_aspect('equal')
        ax_fine_grid(ax)

        # Full Trajectory
        full_traj_color = 'blue'
        full_traj_transparency = 0.1
        full_traj_linestyle = '-'
        ax.plot(self._east_positions, 
                self._north_positions, 
                color = full_traj_color, 
                alpha = full_traj_transparency,
                linestyle = full_traj_linestyle)
        
        # Line to progressively reveal the trajectory
        trace_color = 'blue'
        trace_transparency = 1.0
        trace_linestyle = '--'
        traced_path, = ax.plot([], [], color=trace_color, alpha=trace_transparency, linestyle=trace_linestyle)

        # Current position of the object
        marker = 'ro'
        dot_size = 8
        current_location, = ax.plot([], [], marker, markersize = dot_size)

        # POI
        if poi is not None:
            dot_size = 50
            ax.scatter(poi[1], poi[0], c='orange', marker='x', s=dot_size)

        # Time Slider
        time_slider = self._make_time_slider()

        # Update function - how slider's state affects the values
        def update(val):
            # Get the index corresponding to the current time
            idx = np.searchsorted(self._times, time_slider.val)  # Find the closest index
            
            # Update current location
            current_location.set_data([self._east_positions[idx]], [self._north_positions[idx]])

            # Update the traced path
            traced_path.set_data(self._east_positions[:idx+1], self._north_positions[:idx+1])

            fig.canvas.draw_idle()

        # Connect Slider to update func
        time_slider.on_changed(update)

        plt.show(block=True)

    def three_dim_trajectory(self, n_lim: list = None, e_lim: list = None, d_lim: list = None, title_override: str = None):

        if self._down_available == False:
            raise Exception(f"Unable to generate 3D plot, only 2D data provided.")

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        if title_override is not None:
            ax.set_title(title_override)
        else:
            ax.set_title('3D Trajectory')
        ax.set_xlabel(f'North ({self._position_units})')
        ax.set_ylabel(f'East ({self._position_units})')
        ax.set_zlabel(f'Down ({self._position_units})')
        if d_lim is not None:
            ax.set_zlim(d_lim)
        ax.grid(True)
        ax.view_init(elev=-20, azim=150, roll=180)

        # Full Trajectory
        full_traj_color = 'blue'
        full_traj_transparency = 0.1
        full_traj_linestyle = '-'
        ax.plot(self._north_positions, 
                self._east_positions,
                self._down_positions,
                color = full_traj_color, 
                alpha = full_traj_transparency,
                linestyle = full_traj_linestyle)
        ax.set_aspect('equalxy')
        
        # Line to progressively reveal the trajectory
        trace_color = 'blue'
        trace_transparency = 1.0
        trace_linestyle = '--'
        traced_path, = ax.plot([], [], [], color=trace_color, alpha=trace_transparency, linestyle=trace_linestyle)

        # Current position of the object
        marker = 'ro'
        dot_size = 8
        current_location, = ax.plot([], [], [], marker, markersize = dot_size)

        # Time Slider
        time_slider = self._make_time_slider()

        # Update function - how slider's state affects the values
        def update(val):
            # Get the index corresponding to the current time
            idx = np.searchsorted(self._times, time_slider.val)  # Find the closest index
            
            # Update current location
            current_location.set_data_3d([self._north_positions[idx]], [self._east_positions[idx]], [self._down_positions[idx]])

            # Update the traced path
            traced_path.set_data_3d(self._north_positions[:idx+1], self._east_positions[:idx+1], self._down_positions[:idx+1])

            fig.canvas.draw_idle()

        # Connect Slider to update func
        time_slider.on_changed(update)

        plt.show(block=True)

# EXAMPLE REPLAYER PREP:
# fpath = '/home/natec/MAD/flight_data/20251112_MAD_Flt25_SGM0/1054_2025_11_12_19_14_45_WVM.csv'
"""
fpath = '/mnt/sda1/weak_signals/sensor_data/20251001_MAD_Neenach_Flights/20251001_MAD_Flt16_SGM0/774_2025_10_01_17_41_09_WVM.csv'
flight_data = pd.read_csv(fpath)
flight_data.dropna()
flight_data_trimmed, _ = trim_after_gps(flight_data, sys_time_header='system_time_ns')
flight_data_trimmed = flight_data_trimmed.dropna()
times = flight_data_trimmed['QTFM:Last Timestamp'].to_numpy()
npos = flight_data_trimmed['NED X (m)'].to_numpy()
epos = flight_data_trimmed['NED Y (m)'].to_numpy()
dpos = flight_data_trimmed['NED Z (m)'].to_numpy()
replayer = TrajectoryReplayer(times, npos, epos, dpos, 'ms', 'm')
replayer.planar_trajectory()
"""










