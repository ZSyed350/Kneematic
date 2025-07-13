from nptdms import TdmsFile
import numpy as np

def load_tdms_file(filepath, verbose=False):
    tdms_file = TdmsFile.read(filepath)

    if verbose:
        # List all groups and channels
        for group in tdms_file.groups():
            print(f"Group: {group.name}")
            for channel in group.channels():
                print(f" Channel: {channel.name}")

    return tdms_file

def extract_openknee_data(filepath, verbose=False):
    tdms_file = load_tdms_file(filepath, verbose) 

    # Extract relevant signals
    # NOTE the first 20 samples have odd dt, not sure how that effects the analysis
    flexion_angle = tdms_file["Kinematics.JCS.Actual"]["Flexion Angle"][:]
    extension_torque = tdms_file["State.JCS Load"]["JCS Load Extension Torque"][:]
    actual_dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:] # Control loop Δt values in ms
    # setpoint_time = tdms_file["Timing.Sync Trigger"]["Setpoint Time"][:] # Not sure exactly what this is

    # Time reconstruction
    actual_dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:] # Control loop intervals in ms
    actual_dt_s = actual_dt / 1000.0 # Convert to seconds
    time = np.cumsum(actual_dt_s) # Time signal in seconds

    return time, flexion_angle, extension_torque

if __name__ == "__main__":
    load_tdms_file("OpenKneeData/joint_mechanics-oks003/PatellofemoralJoint/KinematicsKinetics/001_native passive flexion 0 - 60/Data/001_native passive flexion 0 - 60_main_processed_Left.tdms", verbose=True)