SET(CLBLAST_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/clblast/include
  $ENV{CLBLAST_HOME}
  $ENV{CLBLAST_HOME}/include
)

SET(CLBLAST_LIB_SEARCH_PATHS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/clblast/lib
        $ENV{CLBLAST_HOME}
        $ENV{CLBLAST_HOME}/lib
 )

FIND_PATH(CLBLAST_INCLUDE_DIR NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(CLBLAST_LIBRARY NAMES clblast PATHS ${CLBLAST_LIB_SEARCH_PATHS})

SET(CLBLAST_FOUND ON)

#    Check libraries
IF(NOT CLBLAST_LIBRARY)
    SET(CLBLAST_FOUND OFF)
    MESSAGE(STATUS "Could not find CLBlast lib. Turning CLBLAST_FOUND off")
ENDIF()

IF (CLBLAST_FOUND)
  IF (NOT CLBLAST_FIND_QUIETLY)
    MESSAGE(STATUS "Found CLBLAST libraries: ${CLBLAST_LIBRARY}")
    MESSAGE(STATUS "Found CLBLAST include: ${CLBLAST_INCLUDE_DIR}")
  ENDIF (NOT CLBLAST_FIND_QUIETLY)
ELSE (CLBLAST_FOUND)
  IF (CLBLAST_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find CLBLAST")
  ENDIF (CLBLAST_FIND_REQUIRED)
ENDIF (CLBLAST_FOUND)

MARK_AS_ADVANCED(
    CLBLAST_INCLUDE_DIR
    CLBLAST_LIBRARY
)

