"""
This file contains the constants variables and paths used by the rest of the code.
"""

import os

# Path to load flow simulations.
DATA_FOLDER = '/Volumes/Eluteng/EPFL/data'
DATASET_FOLDER = os.path.join(DATA_FOLDER, 'datasets')
_files =  {'cigre13': 'dataset_1_full',
           'cigreLV6bus': 'cigreLV6bus_full',
           'cigreLV6bus_PV': 'cigreLV6buswPV_full',
           'cigreLV6bus_PV_10R': 'cigreLV6buswPV_Rby10_full',
           'cigre4': 'cigreLV4nodewPVRby1',
           'cigre4-Rby5': 'cigreLV4nodewPVRby5',
           'cigre4-Rto5': 'cigreLV4nodewPVRto5',
           'cigre4-Rto2': 'cigreLV4nodewPVRto2'}
DATASET_PATHS = {k: os.path.join(DATA_FOLDER, v) for k, v in _files.items()}

# Sensor classes and corresponding error standard deviations.
SENSOR_CLASSES = [0., 0.1, 0.2, 0.5, 1.]

SENSOR_ABS_MAX_ERROR = {sc: sc / 100. for sc in SENSOR_CLASSES}
SENSOR_STD_ABS = {k: v / 3. for k, v in SENSOR_ABS_MAX_ERROR.items()}  # std = max_error / 3

SENSOR_ANG_MAX_ERROR = {0.: 0., 0.1: 1.5e-3, 0.2: 3e-3, 0.5: 9e-3, 1.: 18e-3}
SENSOR_STD_ANG = {k: v / 3. for k, v in SENSOR_ANG_MAX_ERROR.items()}  # std = max_error / 3
