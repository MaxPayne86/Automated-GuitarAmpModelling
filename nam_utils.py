"""

Based on _calibrate_delay_v1 from https://github.com/sdatkinson/neural-amp-modeler/blob/413d031b92e011ec0b3e6ab3b865b8632725a219/nam/train/core.py#L60
Copyright (c) 2022 Steven Atkinson
SPDX - License - Identifier: MIT

"""

from typing import Dict, Optional, Sequence, Tuple, Union
from pydantic import BaseModel

import numpy as np

# Training using the simplified trainers in NAM is done at 48k.
STANDARD_SAMPLE_RATE = 48_000.0

class _DataInfo(BaseModel):
    """
    :param major_version: Data major version
    """

    major_version: int
    rate: Optional[float]
    t_blips: int
    first_blips_start: int
    t_validate: int
    train_start: int
    validation_start: int
    noise_interval: Tuple[int, int]
    blip_locations: Sequence[Sequence[int]]


_V1_DATA_INFO = _DataInfo(
    major_version=1,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=48_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-432_000,
    noise_interval=(0, 6000),
    blip_locations=((12_000, 36_000),),
)
# V2:
# (0:00-0:02) Blips at 0:00.5 and 0:01.5
# (0:02-0:05) Chirps
# (0:05-0:07) Noise
# (0:07-2:50.5) General training data
# (2:50.5-2:51) Silence
# (2:51-3:00) Validation 1
# (3:00-3:09) Validation 2
# (3:09-3:11) Blips at 3:09.5 and 3:10.5
_V2_DATA_INFO = _DataInfo(
    major_version=2,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-960_000,  # 96_000 + 2 * 432_000
    noise_interval=(12_000, 18_000),
    blip_locations=((24_000, 72_000), (-72_000, -24_000)),
)
# V3:
# (0:00-0:09) Validation 1
# (0:09-0:10) Silence
# (0:10-0:12) Blips at 0:10.5 and 0:11.5
# (0:12-0:15) Chirps
# (0:15-0:17) Noise
# (0:17-3:00.5) General training data
# (3:00.5-3:01) Silence
# (3:01-3:10) Validation 2
_V3_DATA_INFO = _DataInfo(
    major_version=3,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,
    first_blips_start=480_000,
    t_validate=432_000,
    train_start=480_000,
    validation_start=-432_000,
    noise_interval=(492_000, 498_000),
    blip_locations=((504_000, 552_000),),
)
# V4 (aka GuitarML Proteus)
# https://github.com/GuitarML/Releases/releases/download/v1.0.0/Proteus_Capture_Utility.zip
# * 44.1k
# * Odd length...
# * There's a blip on sample zero. This has to be ignored or else over-compensated
#   latencies will come out wrong!
# (0:00-0:01) Blips at 0:00.0 and 0:00.5
# (0:01-0:09) Sine sweeps
# (0:09-0:17) White noise
# (0:17:0.20) Rising white noise (to 0:20.333 appx)
# (0:20-3:30.858) General training data (ends on sample 9,298,872)
# I'm arbitrarily assigning the last 10 seconds as validation data.
_V4_DATA_INFO = _DataInfo(
    major_version=4,
    rate=44_100.0,
    t_blips=44_099,  # Need to ignore the first blip!
    first_blips_start=1,  # Need to ignore the first blip!
    t_validate=441_000,
    # Blips are problematic for training because they don't have preceding silence
    train_start=44_100,
    validation_start=-441_000,
    noise_interval=(6_000, 12_000),
    blip_locations=((22_050,),),
)

_DELAY_CALIBRATION_ABS_THRESHOLD = 0.0003
_DELAY_CALIBRATION_REL_THRESHOLD = 0.001
_DELAY_CALIBRATION_SAFETY_FACTOR = 4

def _calibrate_delay_v_all(
    data_info: _DataInfo,
    y,
    abs_threshold=_DELAY_CALIBRATION_ABS_THRESHOLD,
    rel_threshold=_DELAY_CALIBRATION_REL_THRESHOLD,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
) -> int:
    """
    Calibrate the delay in teh input-output pair based on blips.
    This only uses the blips in the first set of blip locations!

    :param y: The output audio, in complete.
    """
    lookahead = 1_000
    lookback = 10_000
    # Calibrate the trigger:
    y = y[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    background_level = np.max(
        np.abs(
            y[
                data_info.noise_interval[0]
                - data_info.first_blips_start : data_info.noise_interval[1]
                - data_info.first_blips_start
            ]
        )
    )
    trigger_threshold = max(
        background_level + abs_threshold,
        (1.0 + rel_threshold) * background_level,
    )

    delays = []
    for blip_index, i_abs in enumerate(data_info.blip_locations[0], 1):
        # Relative to start of the data
        i_rel = i_abs - data_info.first_blips_start
        start_looking = i_rel - lookahead
        stop_looking = i_rel + lookback
        y_scan = y[start_looking:stop_looking]
        triggered = np.where(np.abs(y_scan) > trigger_threshold)[0]
        if len(triggered) == 0:
            return None
        else:
            j = triggered[0]
            delays.append(j + start_looking - i_rel)

    delay = int(np.min(delays)) - safety_factor

    return delay
