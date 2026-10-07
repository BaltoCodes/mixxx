#[=======================================================================[.rst:
Findsoxr
--------

Finds the SoX Resampler library (libsoxr).

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``soxr::soxr``
  The soxr library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``soxr_FOUND``
  True if the system has the soxr library.
``soxr_INCLUDE_DIRS``
  Include directories needed to use soxr.
``soxr_LIBRARIES``
  Libraries needed to link to soxr.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``soxr_INCLUDE_DIR``
  The directory containing ``soxr.h``.
``soxr_LIBRARY``
  The path to the soxr library.

#]=======================================================================]

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_soxr QUIET soxr)
endif()

find_path(
  soxr_INCLUDE_DIR
  NAMES soxr.h
  HINTS ${PC_soxr_INCLUDE_DIRS}
  DOC "soxr include directory"
)
mark_as_advanced(soxr_INCLUDE_DIR)

find_library(
  soxr_LIBRARY
  NAMES soxr
  HINTS ${PC_soxr_LIBRARY_DIRS}
  DOC "soxr library"
)
mark_as_advanced(soxr_LIBRARY)

if(DEFINED PC_soxr_VERSION AND NOT PC_soxr_VERSION STREQUAL "")
  set(soxr_VERSION "${PC_soxr_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  soxr
  REQUIRED_VARS soxr_LIBRARY soxr_INCLUDE_DIR
  VERSION_VAR soxr_VERSION
)

if(soxr_FOUND)
  set(soxr_LIBRARIES "${soxr_LIBRARY}")
  set(soxr_INCLUDE_DIRS "${soxr_INCLUDE_DIR}")
  set(soxr_DEFINITIONS ${PC_soxr_CFLAGS_OTHER})

  if(NOT TARGET soxr::soxr)
    add_library(soxr::soxr UNKNOWN IMPORTED)
    set_target_properties(
      soxr::soxr
      PROPERTIES
        IMPORTED_LOCATION "${soxr_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${PC_soxr_CFLAGS_OTHER}"
        INTERFACE_INCLUDE_DIRECTORIES "${soxr_INCLUDE_DIR}"
    )
  endif()
endif()
