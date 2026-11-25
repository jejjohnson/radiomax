"""A missing input gives missing angles, never an angle of 0."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from radiomax import geometry


# every input is needed for every output of these
FUNCTIONS = [
    geometry.solar_angles,
    geometry.view_angles,
    geometry.apparent_position,
    geometry.earth_intersection,
    geometry.relative_azimuth,
    geometry.phase_angle,
    geometry.scan_pixel_growth,
]


@pytest.mark.parametrize("function", FUNCTIONS, ids=lambda function: function.__name__)
def test_a_missing_input_gives_missing_outputs(function) -> None:
    required = [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.default is inspect.Parameter.empty
    ]
    values = [np.asarray([20.0 + 3.0 * n, np.nan]) for n in range(len(required))]
    for position in range(len(values)):
        inputs = [np.nan_to_num(value, nan=value[0]) for value in values]
        inputs[position] = values[position]
        outputs = function(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        for output in outputs:
            if np.issubdtype(output.dtype, np.floating):
                assert np.isnan(output[1]), required[position].name
