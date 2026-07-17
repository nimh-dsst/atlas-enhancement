# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.15...3.31)

project( surf_cgal )


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()

endif()

create_single_source_cgal_program( "Flatten_Authalic.cxx" )
create_single_source_cgal_program( "Flatten_Authalic_Iterative.cxx" )
create_single_source_cgal_program( "Nearest_KNN.cxx" )
create_single_source_cgal_program( "Reconstruct_JetAdv.cxx" )
create_single_source_cgal_program( "Reconstruct_PCAdv.cxx" )
create_single_source_cgal_program( "Reconstruct_PCAlpha.cxx" )

option(
  BUILD_CUSTOM_BORDER_PARAMETERIZERS
  "Build parameterizers that require the repository's custom CGAL headers"
  OFF
)
if(BUILD_CUSTOM_BORDER_PARAMETERIZERS)
  create_single_source_cgal_program( "Flatten_Authalic_convex.cxx" )
  create_single_source_cgal_program( "Flatten_Authalic_parametric.cxx" )
  create_single_source_cgal_program( "Flatten_Authalic_Iterative_convex.cxx" )
  create_single_source_cgal_program( "Flatten_Authalic_Iterative_parametric.cxx" )
endif()


# Eigen3 and its components
find_package(Eigen3 3.1.0) #(requires 3.1.0 or greater)
include(CGAL_Eigen3_support)
if(TARGET CGAL::Eigen3_support)

    target_link_libraries(Flatten_Authalic PUBLIC CGAL::Eigen3_support)
    target_link_libraries(Flatten_Authalic_Iterative PUBLIC CGAL::Eigen3_support)
    target_link_libraries(Nearest_KNN PUBLIC CGAL::Eigen3_support)
    target_link_libraries(Reconstruct_JetAdv PUBLIC CGAL::Eigen3_support)
    target_link_libraries(Reconstruct_PCAdv PUBLIC CGAL::Eigen3_support)
    target_link_libraries(Reconstruct_PCAlpha PUBLIC CGAL::Eigen3_support)
    if(BUILD_CUSTOM_BORDER_PARAMETERIZERS)
      target_link_libraries(Flatten_Authalic_convex PUBLIC CGAL::Eigen3_support)
      target_link_libraries(Flatten_Authalic_parametric PUBLIC CGAL::Eigen3_support)
      target_link_libraries(Flatten_Authalic_Iterative_convex PUBLIC CGAL::Eigen3_support)
      target_link_libraries(Flatten_Authalic_Iterative_parametric PUBLIC CGAL::Eigen3_support)
    endif()

else()

  message(STATUS "NOTICE: This project requires Eigen 3.1 (or greater), and will not be compiled.")
  return()

endif()


# OpenMP for Eigen
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Flatten_Authalic PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(Flatten_Authalic_Iterative PUBLIC OpenMP::OpenMP_CXX)
    if(BUILD_CUSTOM_BORDER_PARAMETERIZERS)
      target_link_libraries(Flatten_Authalic_convex PUBLIC OpenMP::OpenMP_CXX)
      target_link_libraries(Flatten_Authalic_parametric PUBLIC OpenMP::OpenMP_CXX)
      target_link_libraries(Flatten_Authalic_Iterative_convex PUBLIC OpenMP::OpenMP_CXX)
      target_link_libraries(Flatten_Authalic_Iterative_parametric PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
