# SPDX-License-Identifier: Apache-2.0
"""Configure the real operator CMake entry without a CANN compiler.

Only external CANN registration helpers are stubbed; the repository's actual
add_op_to_compiled_list macro catches registration at the wrong directory level.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("missing_pto", [False, True])
def test_operator_cmake_registration(tmp_path, missing_pto):
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake.exe")
    ninja = shutil.which("ninja") or str(Path(sys.executable).parent / "ninja.exe")
    if not Path(cmake).exists() or not Path(ninja).exists():
        pytest.skip("install cmake and ninja to check build registration")
    pto_root = tmp_path if missing_pto else ROOT / "csrc"
    source = f'''
cmake_minimum_required(VERSION 3.20)
project(mega_registration NONE)
include("{ROOT.as_posix()}/csrc/cmake/func.cmake")
set(OPS_TRANSFORMER_DIR "{pto_root.as_posix()}")
set(BUILD_OPEN_PROJECT ON)
add_library(op_host_aclnn INTERFACE)
function(add_ops_compile_options)
    cmake_parse_arguments(OPS "" "OP_NAME" "OPTIONS" ${{ARGN}})
    if(NOT OPS_OP_NAME STREQUAL "AscendMegaGdnMtpDecode")
        message(FATAL_ERROR "wrong CANN identity")
    endif()
endfunction()
function(add_modules_sources)
    cmake_parse_arguments(OPS "" "OPTYPE;ACLNNTYPE" "" ${{ARGN}})
    if(NOT OPS_OPTYPE STREQUAL "ascend_mega_gdn_mtp_decode" OR NOT OPS_ACLNNTYPE STREQUAL "aclnn")
        message(FATAL_ERROR "wrong ACLNN registration")
    endif()
endfunction()
add_subdirectory("{ROOT.as_posix()}/csrc/attention/ascend_mega_gdn_mtp_decode" op)
if(NOT COMPILED_OPS STREQUAL "ascend_mega_gdn_mtp_decode")
    message(FATAL_ERROR "wrong compiled op directory: ${{COMPILED_OPS}}")
endif()
'''
    (tmp_path / "CMakeLists.txt").write_text(source, encoding="utf-8")
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build"), "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"],
        capture_output=True,
        text=True,
    )
    if missing_pto:
        assert result.returncode != 0
        assert "pinned PTO submodule" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
