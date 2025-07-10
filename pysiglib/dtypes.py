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

from ctypes import c_float, c_double, c_int32, c_int64
from .load_siglib import cpsig

######################################################
# Some dicts to simplify dtype cases
######################################################

DTYPES = {
    "int32": c_int32,
    "int64": c_int64,
    "float32": c_float,
    "float64": c_double
}

CPSIG_SIGNATURE = {
    "int32": cpsig.signature_int32,
    "int64": cpsig.signature_int64,
    "float32": cpsig.signature_float,
    "float64": cpsig.signature_double
}

CPSIG_BATCH_SIGNATURE = {
    "int32": cpsig.batch_signature_int32,
    "int64": cpsig.batch_signature_int64,
    "float32": cpsig.batch_signature_float,
    "float64": cpsig.batch_signature_double
}