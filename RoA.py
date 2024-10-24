import openhdemg.library as emg
from os.path import join

"""
This script calculates the Rate of Agreement (RoA) between motor units 
identified by two different operators based on their discharge timings.

The RoA is computed for each pair of motor units, comparing the discharges 
from both operators within a specified tolerance. The results are printed 
for each motor unit.

Requirements:
- openhdemg.library: This library is used for loading and processing EMG data.
"""

# Directory with all measurements
MEASUREMENTS_DIR = '.\\RoA_test_files'

# Load data for the two operators
filepath_op_1 = join(
    MEASUREMENTS_DIR,
    'training_20240611_085441_decomp.json'
)
filepath_op_2 = join(
    MEASUREMENTS_DIR,
    'training_20240611_085441_decomp_edited.json'
)

# Load motor unit data for both operators
operator1 = emg.emg_from_json(filepath_op_1)  # list of arrays for operator 1
operator1_indices = emg.sort_mus(operator1)["MUPULSES"]
operator2_indices = emg.emg_from_json(filepath_op_2)["MUPULSES"]  # list of arrays for operator 2

def calculate_roa(motor_unit_op1, motor_unit_op2, tolerance=1):
    """
    Calculate the Rate of Agreement (RoA) for a pair of motor units.

    Args:
        motor_unit_op1 (list): Discharges identified by operator 1.
        motor_unit_op2 (list): Discharges identified by operator 2.
        tolerance (int): The time window (in time points) for matching discharges.

    Returns:
        float: RoA value between 0 and 1.
    """
    A_j = 0  # Discharges identified by both operators within tolerance
    I_j = 0  # Discharges identified by operator 1 but not by operator 2
    S_j = 0  # Discharges identified by operator 2 but not by operator 1

    # Iterate over operator 1's discharges and compare with operator 2
    for discharge in motor_unit_op1:
        if any(abs(discharge - pulse) <= tolerance for pulse in motor_unit_op2):
            A_j += 1  # Found a match within tolerance
        else:
            I_j += 1  # No match found, it's unique to operator 1

    # Iterate over operator 2's discharges and compare with operator 1 (for unique discharges in operator 2)
    for discharge in motor_unit_op2:
        if not any(abs(discharge - pulse) <= tolerance for pulse in motor_unit_op1):
            S_j += 1  # No match found, it's unique to operator 2

    # Calculate RoA
    RoA_j = A_j / (A_j + I_j + S_j) if (A_j + I_j + S_j) > 0 else 0
    return RoA_j

def calculate_roa_for_all_mu(op1_indices, op2_indices, tolerance=1):
    """
    Calculate RoA for all pairs of motor units from two operators.

    Args:
        op1_indices (list): List of motor units from operator 1.
        op2_indices (list): List of motor units from operator 2.
        tolerance (int): The time window (in time points) for matching discharges.

    Returns:
        list: RoA values for each motor unit pair.
    """
    assert len(op1_indices) == len(op2_indices), "Number of motor units must be the same for both operators"

    roa_per_mu = []
    for mu_op1, mu_op2 in zip(op1_indices, op2_indices):
        roa = calculate_roa(mu_op1, mu_op2, tolerance)
        roa_per_mu.append(roa)

    return roa_per_mu

# Calculate RoA for all motor units with a tolerance of ±2 time points
roa_results = calculate_roa_for_all_mu(operator1_indices, operator2_indices, tolerance=2)

# Output the RoA for each motor unit
for i, roa in enumerate(roa_results, 1):
    print(f"Motor Unit {i}: RoA = {roa:.3f}")
