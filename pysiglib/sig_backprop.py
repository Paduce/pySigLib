# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import Union
from ctypes import c_double, POINTER

import numpy as np
import torch

from .error_codes import err_msg

from .load_siglib import cpsig

from .data_handlers import SigDataHandler

def sig_backprop(#WARNING: sig_derivs and sig are non-const here #TODO: fix
        path : Union[np.ndarray, torch.tensor],
        sig_derivs : Union[np.ndarray, torch.tensor],
        sig : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False
) -> Union[np.ndarray, torch.tensor]:
    data = SigDataHandler(path, degree, time_aug, lead_lag)
    out = np.zeros(
        shape=path.shape,
        dtype=np.float64
    )
    err_code = cpsig.sig_backprop_double(
        data.data_ptr,
        out.ctypes.data_as(POINTER(c_double)),
        sig_derivs.ctypes.data_as(POINTER(c_double)),
        sig.ctypes.data_as(POINTER(c_double)),
        data.dimension,
        data.length,
        data.degree,
        time_aug,
        lead_lag
    )

    if err_code:
        raise Exception("Error in pysiglib.signature: " + err_msg(err_code))
    return out