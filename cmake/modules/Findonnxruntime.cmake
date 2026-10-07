#[=======================================================================[.rst:
Findonnxruntime
---------------

Finds the ONNX Runtime library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``onnxruntime::onnxruntime``
  The ONNX Runtime library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``onnxruntime_FOUND``
  True if the system has the ONNX Runtime library.
``onnxruntime_INCLUDE_DIRS``
  Include directories needed to use ONNX Runtime.
``onnxruntime_LIBRARIES``
  Libraries needed to link to ONNX Runtime.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``onnxruntime_INCLUDE_DIR``
  The directory containing ``onnxruntime_cxx_api.h``.
``onnxruntime_LIBRARY``
  The path to the ONNX Runtime library.

#]=======================================================================]

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_onnxruntime QUIET libonnxruntime)
endif()

find_path(
  onnxruntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${PC_onnxruntime_INCLUDE_DIRS}
  PATH_SUFFIXES onnxruntime onnxruntime/core/session
  DOC "ONNX Runtime include directory"
)
mark_as_advanced(onnxruntime_INCLUDE_DIR)

find_library(
  onnxruntime_LIBRARY
  NAMES onnxruntime
  HINTS ${PC_onnxruntime_LIBRARY_DIRS}
  DOC "ONNX Runtime library"
)
mark_as_advanced(onnxruntime_LIBRARY)

if(DEFINED PC_onnxruntime_VERSION AND NOT PC_onnxruntime_VERSION STREQUAL "")
  set(onnxruntime_VERSION "${PC_onnxruntime_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  onnxruntime
  REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR
  VERSION_VAR onnxruntime_VERSION
)

if(onnxruntime_FOUND)
  set(onnxruntime_LIBRARIES "${onnxruntime_LIBRARY}")
  set(onnxruntime_INCLUDE_DIRS "${onnxruntime_INCLUDE_DIR}")

  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
    )
  endif()
endif()
