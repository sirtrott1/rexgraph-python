cmake_minimum_required(VERSION 3.24)
project(rexgraph_cpp_bindings VERSION 0.2.0 LANGUAGES CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  add_compile_options("-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=."
                      "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(RFC 0.3.0 EXACT CONFIG REQUIRED)
get_target_property(rfc_include rfc::numeric INTERFACE_INCLUDE_DIRECTORIES)
list(GET rfc_include 0 rfc_include)
file(GLOB_RECURSE rfc_headers CONFIGURE_DEPENDS "${rfc_include}/rfc/*.hpp")
list(SORT rfc_headers)
set(rfc_receipt "")
foreach(header IN LISTS rfc_headers)
  file(RELATIVE_PATH relative "${rfc_include}" "${header}")
  file(SHA256 "${header}" digest)
  string(APPEND rfc_receipt "${relative}:${digest}\n")
endforeach()
string(SHA256 rfc_digest "${rfc_receipt}")
if(NOT rfc_headers)
  message(FATAL_ERROR "An installed RFC SDK is required")
endif()

# A standalone bridge source archive carries the same package beside this file.
# The repository keeps its authoritative Python package at the top level.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rexgraph_cpp/__init__.py")
  set(rexgraph_python_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/rexgraph_cpp")
else()
  set(rexgraph_python_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../rexgraph_cpp")
endif()
if(NOT EXISTS "${rexgraph_python_source_dir}/__init__.py")
  message(FATAL_ERROR "The rexgraph_cpp Python package is missing")
endif()

function(rexgraph_binding_receipt output)
  set(receipt "")
  foreach(source IN LISTS ARGN)
    if(IS_ABSOLUTE "${source}")
      set(source_path "${source}")
      file(RELATIVE_PATH relative "${rexgraph_python_source_dir}" "${source}")
      if(relative MATCHES "^\\.\\.")
        message(FATAL_ERROR "Binding source lies outside its declared package")
      endif()
      set(source_name "rexgraph_cpp/${relative}")
    else()
      set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
      set(source_name "${source}")
    endif()
    file(SHA256 "${source_path}" digest)
    string(APPEND receipt "${source_name}:${digest}\n")
  endforeach()
  string(SHA256 digest "${receipt}")
  set(${output} "${digest}" PARENT_SCOPE)
endfunction()

set(binding_sources include/bridge.hpp src/bridge.cpp ${rexgraph_python_source_dir}/_bridge.pyx ${rexgraph_python_source_dir}/_declaration.py
  include/algebraic_bridge.hpp src/algebraic_bridge.cpp include/phase_fold_bridge.hpp src/phase_fold_bridge.cpp
  include/chain_bridge.hpp src/chain_bridge.cpp)
rexgraph_binding_receipt(binding_digest ${binding_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${rfc_headers} ${binding_sources})
set(cython_output "${CMAKE_CURRENT_BINARY_DIR}/_bridge.cpp")
add_custom_command(OUTPUT "${cython_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3
    --output-file "${cython_output}" "${rexgraph_python_source_dir}/_bridge.pyx"
  DEPENDS ${rexgraph_python_source_dir}/_bridge.pyx include/bridge.hpp include/algebraic_bridge.hpp include/phase_fold_bridge.hpp include/chain_bridge.hpp VERBATIM)
Python3_add_library(_bridge MODULE WITH_SOABI "${cython_output}" src/bridge.cpp src/algebraic_bridge.cpp src/phase_fold_bridge.cpp src/chain_bridge.cpp)
target_include_directories(_bridge PRIVATE include)
target_link_libraries(_bridge PRIVATE rfc::inference rfc::learning rfc::cpu_tiles rfc::storage rfc::native_query_execution)
target_compile_definitions(_bridge PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_BINDING_DIGEST="${binding_digest}")
install(TARGETS _bridge LIBRARY DESTINATION rexgraph_cpp)

set(binary_sources include/binary_bridge.hpp src/binary_bridge.cpp ${rexgraph_python_source_dir}/_binary.pyx)
rexgraph_binding_receipt(binary_digest ${binary_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${binary_sources})
set(binary_cython_output "${CMAKE_CURRENT_BINARY_DIR}/_binary.cpp")
add_custom_command(OUTPUT "${binary_cython_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${binary_cython_output}"
    "${rexgraph_python_source_dir}/_binary.pyx"
  DEPENDS ${rexgraph_python_source_dir}/_binary.pyx include/binary_bridge.hpp VERBATIM)
Python3_add_library(_binary MODULE WITH_SOABI "${binary_cython_output}" src/binary_bridge.cpp)
target_include_directories(_binary PRIVATE include)
target_link_libraries(_binary PRIVATE rfc::learning)
target_compile_definitions(_binary PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_BINARY_BINDING_DIGEST="${binary_digest}")
install(TARGETS _binary LIBRARY DESTINATION rexgraph_cpp)

set(relation_sources include/relational_bridge.hpp ${rexgraph_python_source_dir}/_relation.pyx)
rexgraph_binding_receipt(relation_digest ${relation_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${relation_sources})
set(relation_output "${CMAKE_CURRENT_BINARY_DIR}/_relation.cpp")
add_custom_command(OUTPUT "${relation_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${relation_output}"
    "${rexgraph_python_source_dir}/_relation.pyx"
  DEPENDS ${relation_sources} VERBATIM)
Python3_add_library(_relation MODULE WITH_SOABI "${relation_output}")
target_include_directories(_relation PRIVATE include)
target_link_libraries(_relation PRIVATE rfc::local rfc::storage)
target_compile_definitions(_relation PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_RELATION_BINDING_DIGEST="${relation_digest}")
install(TARGETS _relation LIBRARY DESTINATION rexgraph_cpp)

set(channel_sources include/channel_bridge.hpp ${rexgraph_python_source_dir}/_channels.pyx)
rexgraph_binding_receipt(channel_digest ${channel_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${channel_sources})
set(channel_output "${CMAKE_CURRENT_BINARY_DIR}/_channels.cpp")
add_custom_command(OUTPUT "${channel_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${channel_output}"
    "${rexgraph_python_source_dir}/_channels.pyx"
  DEPENDS ${channel_sources} VERBATIM)
Python3_add_library(_channels MODULE WITH_SOABI "${channel_output}")
target_include_directories(_channels PRIVATE include)
target_link_libraries(_channels PRIVATE rfc::field rfc::storage)
target_compile_definitions(_channels PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_CHANNEL_BINDING_DIGEST="${channel_digest}")
install(TARGETS _channels LIBRARY DESTINATION rexgraph_cpp)

set(stalk_sources include/stalk_field_bridge.hpp ${rexgraph_python_source_dir}/_stalk_fields.pyx)
rexgraph_binding_receipt(stalk_digest ${stalk_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${stalk_sources})
set(stalk_output "${CMAKE_CURRENT_BINARY_DIR}/_stalk_fields.cpp")
add_custom_command(OUTPUT "${stalk_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${stalk_output}"
    "${rexgraph_python_source_dir}/_stalk_fields.pyx"
  DEPENDS ${stalk_sources} VERBATIM)
Python3_add_library(_stalk_fields MODULE WITH_SOABI "${stalk_output}")
target_include_directories(_stalk_fields PRIVATE include)
target_link_libraries(_stalk_fields PRIVATE rfc::local rfc::storage)
target_compile_definitions(_stalk_fields PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_STALK_BINDING_DIGEST="${stalk_digest}")
install(TARGETS _stalk_fields LIBRARY DESTINATION rexgraph_cpp)

set(stalk_boundary_sources include/stalk_boundary_bridge.hpp include/graded_transport_bridge.hpp ${rexgraph_python_source_dir}/_stalk_boundary.pyx ${rexgraph_python_source_dir}/_stalk_fields.pyx)
rexgraph_binding_receipt(stalk_boundary_digest ${stalk_boundary_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${stalk_boundary_sources})
set(stalk_boundary_output "${CMAKE_CURRENT_BINARY_DIR}/_stalk_boundary.cpp")
add_custom_command(OUTPUT "${stalk_boundary_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${stalk_boundary_output}"
    "${rexgraph_python_source_dir}/_stalk_boundary.pyx"
  DEPENDS ${stalk_boundary_sources} VERBATIM)
Python3_add_library(_stalk_boundary MODULE WITH_SOABI "${stalk_boundary_output}")
target_include_directories(_stalk_boundary PRIVATE include)
target_link_libraries(_stalk_boundary PRIVATE rfc::learning rfc::storage)
target_compile_definitions(_stalk_boundary PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_STALK_BOUNDARY_BINDING_DIGEST="${stalk_boundary_digest}")
install(TARGETS _stalk_boundary LIBRARY DESTINATION rexgraph_cpp)

set(sections_sources include/sections_bridge.hpp ${rexgraph_python_source_dir}/_sections.pyx)
rexgraph_binding_receipt(sections_digest ${sections_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${sections_sources})
set(sections_output "${CMAKE_CURRENT_BINARY_DIR}/_sections.cpp")
add_custom_command(OUTPUT "${sections_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${sections_output}"
    "${rexgraph_python_source_dir}/_sections.pyx"
  DEPENDS ${sections_sources} VERBATIM)
Python3_add_library(_sections MODULE WITH_SOABI "${sections_output}")
target_include_directories(_sections PRIVATE include)
target_link_libraries(_sections PRIVATE rfc::local rfc::storage)
target_compile_definitions(_sections PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_SECTIONS_BINDING_DIGEST="${sections_digest}")
install(TARGETS _sections LIBRARY DESTINATION rexgraph_cpp)

set(guard_sources include/guard_bridge.hpp include/lexical_routes_bridge.hpp ${rexgraph_python_source_dir}/_guards.pyx)
rexgraph_binding_receipt(guard_digest ${guard_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${guard_sources})
set(guard_output "${CMAKE_CURRENT_BINARY_DIR}/_guards.cpp")
add_custom_command(OUTPUT "${guard_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${guard_output}"
    "${rexgraph_python_source_dir}/_guards.pyx"
  DEPENDS ${guard_sources} VERBATIM)
Python3_add_library(_guards MODULE WITH_SOABI "${guard_output}")
target_include_directories(_guards PRIVATE include)
target_link_libraries(_guards PRIVATE rfc::storage)
target_compile_definitions(_guards PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_GUARD_BINDING_DIGEST="${guard_digest}")
install(TARGETS _guards LIBRARY DESTINATION rexgraph_cpp)

set(stalk_transport_sources include/stalk_transport_bridge.hpp include/stalk_pattern_bridge.hpp include/stalk_pattern_bank_bridge.hpp include/stalk_objective_bridge.hpp include/stalk_parameter_relations_bridge.hpp include/stalk_boundary_bridge.hpp ${rexgraph_python_source_dir}/_stalk_transport.pyx ${rexgraph_python_source_dir}/_stalk_fields.pyx ${rexgraph_python_source_dir}/_context_views.py)
rexgraph_binding_receipt(stalk_transport_digest ${stalk_transport_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${stalk_transport_sources})
set(stalk_transport_output "${CMAKE_CURRENT_BINARY_DIR}/_stalk_transport.cpp")
add_custom_command(OUTPUT "${stalk_transport_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${stalk_transport_output}"
    "${rexgraph_python_source_dir}/_stalk_transport.pyx"
  DEPENDS ${stalk_transport_sources} VERBATIM)
Python3_add_library(_stalk_transport MODULE WITH_SOABI "${stalk_transport_output}")
target_include_directories(_stalk_transport PRIVATE include)
target_link_libraries(_stalk_transport PRIVATE rfc::learning rfc::storage)
target_compile_definitions(_stalk_transport PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_STALK_TRANSPORT_BINDING_DIGEST="${stalk_transport_digest}")
install(TARGETS _stalk_transport LIBRARY DESTINATION rexgraph_cpp)

set(model_storage_sources include/model_storage_bridge.hpp ${rexgraph_python_source_dir}/_model_storage.pyx ${rexgraph_python_source_dir}/_value_codec.pxi)
rexgraph_binding_receipt(model_storage_digest ${model_storage_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${model_storage_sources})
set(model_storage_output "${CMAKE_CURRENT_BINARY_DIR}/_model_storage.cpp")
add_custom_command(OUTPUT "${model_storage_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3
    --working "${rexgraph_python_source_dir}/.." --output-file "${model_storage_output}"
    "${rexgraph_python_source_dir}/_model_storage.pyx"
  DEPENDS ${model_storage_sources} VERBATIM)
Python3_add_library(_model_storage MODULE WITH_SOABI "${model_storage_output}")
target_include_directories(_model_storage PRIVATE include)
target_link_libraries(_model_storage PRIVATE rfc::storage)
target_compile_definitions(_model_storage PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_MODEL_STORAGE_BINDING_DIGEST="${model_storage_digest}")
install(TARGETS _model_storage LIBRARY DESTINATION rexgraph_cpp)

set(field_program_sources include/field_program_bridge.hpp ${rexgraph_python_source_dir}/_field_program.pyx
  ${rexgraph_python_source_dir}/_stalk_fields.pyx)
rexgraph_binding_receipt(field_program_digest ${field_program_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_program_sources})
set(field_program_output "${CMAKE_CURRENT_BINARY_DIR}/_field_program.cpp")
add_custom_command(OUTPUT "${field_program_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_program_output}"
    "${rexgraph_python_source_dir}/_field_program.pyx"
  DEPENDS ${field_program_sources} VERBATIM)
Python3_add_library(_field_program MODULE WITH_SOABI "${field_program_output}")
target_include_directories(_field_program PRIVATE include)
target_link_libraries(_field_program PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_field_program PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_PROGRAM_BINDING_DIGEST="${field_program_digest}")
install(TARGETS _field_program LIBRARY DESTINATION rexgraph_cpp)

set(field_interface_sources include/field_interface_bridge.hpp include/literal_input_bridge.hpp include/field_program_bridge.hpp
  ${rexgraph_python_source_dir}/_field_interface.pyx ${rexgraph_python_source_dir}/_field_interface.pxd
  ${rexgraph_python_source_dir}/_field_program.pyx ${rexgraph_python_source_dir}/_stalk_fields.pyx)
rexgraph_binding_receipt(field_interface_digest ${field_interface_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_interface_sources})
set(field_interface_output "${CMAKE_CURRENT_BINARY_DIR}/_field_interface.cpp")
add_custom_command(OUTPUT "${field_interface_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_interface_output}"
    "${rexgraph_python_source_dir}/_field_interface.pyx"
  DEPENDS ${field_interface_sources} VERBATIM)
Python3_add_library(_field_interface MODULE WITH_SOABI "${field_interface_output}")
target_include_directories(_field_interface PRIVATE include)
target_link_libraries(_field_interface PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_field_interface PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_INTERFACE_BINDING_DIGEST="${field_interface_digest}")
install(TARGETS _field_interface LIBRARY DESTINATION rexgraph_cpp)

set(field_evidence_sources include/occurrence_target_bridge.hpp include/field_family_bridge.hpp include/field_evidence_bridge.hpp ${field_interface_sources}
  ${rexgraph_python_source_dir}/_field_evidence.pyx ${rexgraph_python_source_dir}/_field_evidence.pxd)
rexgraph_binding_receipt(field_evidence_digest ${field_evidence_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_evidence_sources})
set(field_evidence_output "${CMAKE_CURRENT_BINARY_DIR}/_field_evidence.cpp")
add_custom_command(OUTPUT "${field_evidence_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_evidence_output}"
    "${rexgraph_python_source_dir}/_field_evidence.pyx"
  DEPENDS ${field_evidence_sources} VERBATIM)
Python3_add_library(_field_evidence MODULE WITH_SOABI "${field_evidence_output}")
target_include_directories(_field_evidence PRIVATE include)
target_link_libraries(_field_evidence PRIVATE rfc::learning rfc::storage)
target_compile_definitions(_field_evidence PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_EVIDENCE_BINDING_DIGEST="${field_evidence_digest}")
install(TARGETS _field_evidence LIBRARY DESTINATION rexgraph_cpp)

set(field_comparison_sources include/field_comparison_bridge.hpp ${field_interface_sources}
  ${rexgraph_python_source_dir}/_field_comparison.pyx)
rexgraph_binding_receipt(field_comparison_digest ${field_comparison_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_comparison_sources})
set(field_comparison_output "${CMAKE_CURRENT_BINARY_DIR}/_field_comparison.cpp")
add_custom_command(OUTPUT "${field_comparison_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_comparison_output}"
    "${rexgraph_python_source_dir}/_field_comparison.pyx"
  DEPENDS ${field_comparison_sources} VERBATIM)
Python3_add_library(_field_comparison MODULE WITH_SOABI "${field_comparison_output}")
target_include_directories(_field_comparison PRIVATE include)
target_link_libraries(_field_comparison PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_field_comparison PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_COMPARISON_BINDING_DIGEST="${field_comparison_digest}")
install(TARGETS _field_comparison LIBRARY DESTINATION rexgraph_cpp)

set(field_admission_sources include/field_admission_bridge.hpp ${field_interface_sources}
  ${rexgraph_python_source_dir}/_field_admission.pyx ${rexgraph_python_source_dir}/_field_admission.pxd)
rexgraph_binding_receipt(field_admission_digest ${field_admission_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_admission_sources})
set(field_admission_output "${CMAKE_CURRENT_BINARY_DIR}/_field_admission.cpp")
add_custom_command(OUTPUT "${field_admission_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_admission_output}"
    "${rexgraph_python_source_dir}/_field_admission.pyx"
  DEPENDS ${field_admission_sources} VERBATIM)
Python3_add_library(_field_admission MODULE WITH_SOABI "${field_admission_output}")
target_include_directories(_field_admission PRIVATE include)
target_link_libraries(_field_admission PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_field_admission PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_ADMISSION_BINDING_DIGEST="${field_admission_digest}")
install(TARGETS _field_admission LIBRARY DESTINATION rexgraph_cpp)

set(field_evidence_admission_sources include/field_evidence_admission_bridge.hpp ${field_evidence_sources}
  ${field_admission_sources} ${rexgraph_python_source_dir}/_field_evidence_admission.pyx ${rexgraph_python_source_dir}/_field_evidence_admission.pxd)
rexgraph_binding_receipt(field_evidence_admission_digest ${field_evidence_admission_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_evidence_admission_sources})
set(field_evidence_admission_output "${CMAKE_CURRENT_BINARY_DIR}/_field_evidence_admission.cpp")
add_custom_command(OUTPUT "${field_evidence_admission_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_evidence_admission_output}"
    "${rexgraph_python_source_dir}/_field_evidence_admission.pyx"
  DEPENDS ${field_evidence_admission_sources} VERBATIM)
Python3_add_library(_field_evidence_admission MODULE WITH_SOABI "${field_evidence_admission_output}")
target_include_directories(_field_evidence_admission PRIVATE include)
target_link_libraries(_field_evidence_admission PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_field_evidence_admission PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_EVIDENCE_ADMISSION_BINDING_DIGEST="${field_evidence_admission_digest}")
install(TARGETS _field_evidence_admission LIBRARY DESTINATION rexgraph_cpp)

option(REXGRAPH_BINDING_TEST_FIXTURE "Build the native test fixture writer" OFF)
if(REXGRAPH_BINDING_TEST_FIXTURE)
  add_executable(algebraic_grade_fixture tests/make_algebraic_grade_fixture.cpp)
  target_link_libraries(algebraic_grade_fixture PRIVATE rfc::storage)
  add_executable(binding_fixture tests/make_fixture.cpp)
  target_link_libraries(binding_fixture PRIVATE rfc::inference rfc::learning)
  add_executable(algebraic_fixture tests/make_algebraic_fixture.cpp)
  target_link_libraries(algebraic_fixture PRIVATE rfc::storage)
  add_executable(algebraic_query_fixture tests/make_algebraic_query_fixture.cpp)
  target_link_libraries(algebraic_query_fixture PRIVATE rfc::native_query_execution)
  add_executable(algebraic_graph_history_fixture tests/make_algebraic_graph_history_fixture.cpp)
  target_link_libraries(algebraic_graph_history_fixture PRIVATE rfc::storage)
  add_executable(algebraic_history_fixture tests/make_algebraic_history_fixture.cpp)
  target_link_libraries(algebraic_history_fixture PRIVATE rfc::storage)
  add_executable(phase_fold_fixture tests/make_phase_fold_fixture.cpp)
  target_link_libraries(phase_fold_fixture PRIVATE rfc::storage)
  add_executable(algebraic_graph_fixture tests/make_algebraic_graph_fixture.cpp)
  target_link_libraries(algebraic_graph_fixture PRIVATE rfc::storage)
endif()

set(field_local_sources include/field_local_bridge.hpp include/field_extension_bridge.hpp ${field_evidence_admission_sources}
  ${rexgraph_python_source_dir}/_field_extension.pyx ${rexgraph_python_source_dir}/_field_extension.pxd ${rexgraph_python_source_dir}/_field_local.pyx)
rexgraph_binding_receipt(field_local_digest ${field_local_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_local_sources})
set(field_local_output "${CMAKE_CURRENT_BINARY_DIR}/_field_local.cpp")
add_custom_command(OUTPUT "${field_local_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_local_output}"
    "${rexgraph_python_source_dir}/_field_local.pyx"
  DEPENDS ${field_local_sources} VERBATIM)
Python3_add_library(_field_local MODULE WITH_SOABI "${field_local_output}")
target_include_directories(_field_local PRIVATE include)
target_link_libraries(_field_local PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_field_local PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_LOCAL_BINDING_DIGEST="${field_local_digest}")
install(TARGETS _field_local LIBRARY DESTINATION rexgraph_cpp)

set(field_extension_sources include/field_extension_bridge.hpp ${field_evidence_admission_sources}
  ${rexgraph_python_source_dir}/_field_extension.pyx ${rexgraph_python_source_dir}/_field_extension.pxd)
rexgraph_binding_receipt(field_extension_digest ${field_extension_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${field_extension_sources})
set(field_extension_output "${CMAKE_CURRENT_BINARY_DIR}/_field_extension.cpp")
add_custom_command(OUTPUT "${field_extension_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${field_extension_output}"
    "${rexgraph_python_source_dir}/_field_extension.pyx"
  DEPENDS ${field_extension_sources} VERBATIM)
Python3_add_library(_field_extension MODULE WITH_SOABI "${field_extension_output}")
target_include_directories(_field_extension PRIVATE include)
target_link_libraries(_field_extension PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_field_extension PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FIELD_EXTENSION_BINDING_DIGEST="${field_extension_digest}")
install(TARGETS _field_extension LIBRARY DESTINATION rexgraph_cpp)

set(finite_field_sources include/finite_field_bridge.hpp ${field_evidence_sources}
  ${rexgraph_python_source_dir}/_finite_field.pyx ${rexgraph_python_source_dir}/_finite_field.pxd)
rexgraph_binding_receipt(finite_field_digest ${finite_field_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${finite_field_sources})
set(finite_field_output "${CMAKE_CURRENT_BINARY_DIR}/_finite_field.cpp")
add_custom_command(OUTPUT "${finite_field_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${finite_field_output}"
    "${rexgraph_python_source_dir}/_finite_field.pyx"
  DEPENDS ${finite_field_sources} VERBATIM)
Python3_add_library(_finite_field MODULE WITH_SOABI "${finite_field_output}")
target_include_directories(_finite_field PRIVATE include)
target_link_libraries(_finite_field PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_finite_field PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FINITE_FIELD_BINDING_DIGEST="${finite_field_digest}")
install(TARGETS _finite_field LIBRARY DESTINATION rexgraph_cpp)

set(finite_field_admission_sources include/finite_field_admission_bridge.hpp ${finite_field_sources} ${field_evidence_admission_sources}
  ${rexgraph_python_source_dir}/_finite_field_admission.pyx ${rexgraph_python_source_dir}/_finite_field_admission.pxd)
rexgraph_binding_receipt(finite_field_admission_digest ${finite_field_admission_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${finite_field_admission_sources})
set(finite_field_admission_output "${CMAKE_CURRENT_BINARY_DIR}/_finite_field_admission.cpp")
add_custom_command(OUTPUT "${finite_field_admission_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${finite_field_admission_output}"
    "${rexgraph_python_source_dir}/_finite_field_admission.pyx"
  DEPENDS ${finite_field_admission_sources} VERBATIM)
Python3_add_library(_finite_field_admission MODULE WITH_SOABI "${finite_field_admission_output}")
target_include_directories(_finite_field_admission PRIVATE include)
target_link_libraries(_finite_field_admission PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_finite_field_admission PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FINITE_FIELD_ADMISSION_BINDING_DIGEST="${finite_field_admission_digest}")
install(TARGETS _finite_field_admission LIBRARY DESTINATION rexgraph_cpp)

set(finite_operator_sources include/finite_operator_bridge.hpp ${finite_field_sources}
  ${rexgraph_python_source_dir}/_finite_operator.pyx ${rexgraph_python_source_dir}/_finite_operator.pxd)
rexgraph_binding_receipt(finite_operator_digest ${finite_operator_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${finite_operator_sources})
set(finite_operator_output "${CMAKE_CURRENT_BINARY_DIR}/_finite_operator.cpp")
add_custom_command(OUTPUT "${finite_operator_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${finite_operator_output}"
    "${rexgraph_python_source_dir}/_finite_operator.pyx"
  DEPENDS ${finite_operator_sources} VERBATIM)
Python3_add_library(_finite_operator MODULE WITH_SOABI "${finite_operator_output}")
target_include_directories(_finite_operator PRIVATE include)
target_link_libraries(_finite_operator PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_finite_operator PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FINITE_OPERATOR_BINDING_DIGEST="${finite_operator_digest}")
install(TARGETS _finite_operator LIBRARY DESTINATION rexgraph_cpp)


set(finite_operator_admission_sources include/finite_operator_admission_bridge.hpp ${finite_operator_sources} ${finite_field_admission_sources}
  ${rexgraph_python_source_dir}/_finite_operator_admission.pyx ${rexgraph_python_source_dir}/_finite_operator_admission.pxd)
rexgraph_binding_receipt(finite_operator_admission_digest ${finite_operator_admission_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${finite_operator_admission_sources})
set(finite_operator_admission_output "${CMAKE_CURRENT_BINARY_DIR}/_finite_operator_admission.cpp")
add_custom_command(OUTPUT "${finite_operator_admission_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${finite_operator_admission_output}"
    "${rexgraph_python_source_dir}/_finite_operator_admission.pyx"
  DEPENDS ${finite_operator_admission_sources} VERBATIM)
Python3_add_library(_finite_operator_admission MODULE WITH_SOABI "${finite_operator_admission_output}")
target_include_directories(_finite_operator_admission PRIVATE include)
target_link_libraries(_finite_operator_admission PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_finite_operator_admission PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FINITE_OPERATOR_ADMISSION_BINDING_DIGEST="${finite_operator_admission_digest}")
install(TARGETS _finite_operator_admission LIBRARY DESTINATION rexgraph_cpp)

set(known_tensor_square_sources include/known_tensor_square_bridge.hpp ${field_admission_sources}
  ${rexgraph_python_source_dir}/_known_tensor_square.pyx ${rexgraph_python_source_dir}/_known_tensor_square.pxd)
rexgraph_binding_receipt(known_tensor_square_digest ${known_tensor_square_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${known_tensor_square_sources})
set(known_tensor_square_output "${CMAKE_CURRENT_BINARY_DIR}/_known_tensor_square.cpp")
add_custom_command(OUTPUT "${known_tensor_square_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${known_tensor_square_output}"
    "${rexgraph_python_source_dir}/_known_tensor_square.pyx"
  DEPENDS ${known_tensor_square_sources} VERBATIM)
Python3_add_library(_known_tensor_square MODULE WITH_SOABI "${known_tensor_square_output}")
target_include_directories(_known_tensor_square PRIVATE include)
target_link_libraries(_known_tensor_square PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_known_tensor_square PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_KNOWN_TENSOR_SQUARE_BINDING_DIGEST="${known_tensor_square_digest}")
install(TARGETS _known_tensor_square LIBRARY DESTINATION rexgraph_cpp)

set(known_product_learning_sources include/known_product_learning_bridge.hpp include/four_use_role_application_bridge.hpp ${known_tensor_square_sources}
  ${field_evidence_admission_sources} ${rexgraph_python_source_dir}/_known_product_learning.pyx ${rexgraph_python_source_dir}/_known_product_learning.pxd)
rexgraph_binding_receipt(known_product_learning_digest ${known_product_learning_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${known_product_learning_sources})
set(known_product_learning_output "${CMAKE_CURRENT_BINARY_DIR}/_known_product_learning.cpp")
add_custom_command(OUTPUT "${known_product_learning_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${known_product_learning_output}"
    "${rexgraph_python_source_dir}/_known_product_learning.pyx"
  DEPENDS ${known_product_learning_sources} VERBATIM)
Python3_add_library(_known_product_learning MODULE WITH_SOABI "${known_product_learning_output}")
target_include_directories(_known_product_learning PRIVATE include)
target_link_libraries(_known_product_learning PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_known_product_learning PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_KNOWN_PRODUCT_LEARNING_BINDING_DIGEST="${known_product_learning_digest}")
install(TARGETS _known_product_learning LIBRARY DESTINATION rexgraph_cpp)

set(literal_finite_field_sources include/literal_finite_field_bridge.hpp ${finite_field_sources}
  ${rexgraph_python_source_dir}/_literal_finite_field.pyx ${rexgraph_python_source_dir}/_literal_finite_field.pxd)
rexgraph_binding_receipt(literal_finite_field_digest ${literal_finite_field_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${literal_finite_field_sources})
set(literal_finite_field_output "${CMAKE_CURRENT_BINARY_DIR}/_literal_finite_field.cpp")
add_custom_command(OUTPUT "${literal_finite_field_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${literal_finite_field_output}"
    "${rexgraph_python_source_dir}/_literal_finite_field.pyx"
  DEPENDS ${literal_finite_field_sources} VERBATIM)
Python3_add_library(_literal_finite_field MODULE WITH_SOABI "${literal_finite_field_output}")
target_include_directories(_literal_finite_field PRIVATE include)
target_link_libraries(_literal_finite_field PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_literal_finite_field PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_LITERAL_FINITE_FIELD_BINDING_DIGEST="${literal_finite_field_digest}")
install(TARGETS _literal_finite_field LIBRARY DESTINATION rexgraph_cpp)

set(available_literal_finite_field_sources include/available_literal_finite_field_bridge.hpp ${literal_finite_field_sources} ${finite_field_admission_sources}
  ${rexgraph_python_source_dir}/_available_literal_finite_field.pyx ${rexgraph_python_source_dir}/_available_literal_finite_field.pxd)
rexgraph_binding_receipt(available_literal_finite_field_digest ${available_literal_finite_field_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${available_literal_finite_field_sources})
set(available_literal_finite_field_output "${CMAKE_CURRENT_BINARY_DIR}/_available_literal_finite_field.cpp")
add_custom_command(OUTPUT "${available_literal_finite_field_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${available_literal_finite_field_output}"
    "${rexgraph_python_source_dir}/_available_literal_finite_field.pyx"
  DEPENDS ${available_literal_finite_field_sources} VERBATIM)
Python3_add_library(_available_literal_finite_field MODULE WITH_SOABI "${available_literal_finite_field_output}")
target_include_directories(_available_literal_finite_field PRIVATE include)
target_link_libraries(_available_literal_finite_field PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_available_literal_finite_field PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_AVAILABLE_LITERAL_FINITE_FIELD_BINDING_DIGEST="${available_literal_finite_field_digest}")
install(TARGETS _available_literal_finite_field LIBRARY DESTINATION rexgraph_cpp)

set(known_tensor_graph_sources include/known_tensor_graph_bridge.hpp ${field_interface_sources}
  ${rexgraph_python_source_dir}/_known_tensor_graph.pyx ${rexgraph_python_source_dir}/_known_tensor_graph.pxd)
rexgraph_binding_receipt(known_tensor_graph_digest ${known_tensor_graph_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${known_tensor_graph_sources})
set(known_tensor_graph_output "${CMAKE_CURRENT_BINARY_DIR}/_known_tensor_graph.cpp")
add_custom_command(OUTPUT "${known_tensor_graph_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${known_tensor_graph_output}"
    "${rexgraph_python_source_dir}/_known_tensor_graph.pyx"
  DEPENDS ${known_tensor_graph_sources} VERBATIM)
Python3_add_library(_known_tensor_graph MODULE WITH_SOABI "${known_tensor_graph_output}")
target_include_directories(_known_tensor_graph PRIVATE include)
target_link_libraries(_known_tensor_graph PRIVATE rfc::inference rfc::storage)
target_compile_definitions(_known_tensor_graph PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_KNOWN_TENSOR_GRAPH_BINDING_DIGEST="${known_tensor_graph_digest}")
install(TARGETS _known_tensor_graph LIBRARY DESTINATION rexgraph_cpp)

set(literal_field_family_sources include/literal_field_family_bridge.hpp ${known_tensor_graph_sources} ${field_evidence_sources}
  ${rexgraph_python_source_dir}/_literal_field_family.pyx ${rexgraph_python_source_dir}/_literal_field_family.pxd)
rexgraph_binding_receipt(literal_field_family_digest ${literal_field_family_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${literal_field_family_sources})
set(literal_field_family_output "${CMAKE_CURRENT_BINARY_DIR}/_literal_field_family.cpp")
add_custom_command(OUTPUT "${literal_field_family_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${literal_field_family_output}"
    "${rexgraph_python_source_dir}/_literal_field_family.pyx"
  DEPENDS ${literal_field_family_sources} VERBATIM)
Python3_add_library(_literal_field_family MODULE WITH_SOABI "${literal_field_family_output}")
target_include_directories(_literal_field_family PRIVATE include)
target_link_libraries(_literal_field_family PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_literal_field_family PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_LITERAL_FIELD_FAMILY_BINDING_DIGEST="${literal_field_family_digest}")
install(TARGETS _literal_field_family LIBRARY DESTINATION rexgraph_cpp)

set(available_literal_field_family_sources include/available_literal_field_family_bridge.hpp ${literal_field_family_sources}
  ${field_evidence_admission_sources} ${rexgraph_python_source_dir}/_available_literal_field_family.pyx
  ${rexgraph_python_source_dir}/_available_literal_field_family.pxd)
rexgraph_binding_receipt(available_literal_field_family_digest ${available_literal_field_family_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${available_literal_field_family_sources})
set(available_literal_field_family_output "${CMAKE_CURRENT_BINARY_DIR}/_available_literal_field_family.cpp")
add_custom_command(OUTPUT "${available_literal_field_family_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${available_literal_field_family_output}"
    "${rexgraph_python_source_dir}/_available_literal_field_family.pyx"
  DEPENDS ${available_literal_field_family_sources} VERBATIM)
Python3_add_library(_available_literal_field_family MODULE WITH_SOABI "${available_literal_field_family_output}")
target_include_directories(_available_literal_field_family PRIVATE include)
target_link_libraries(_available_literal_field_family PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_available_literal_field_family PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_AVAILABLE_LITERAL_FIELD_FAMILY_BINDING_DIGEST="${available_literal_field_family_digest}")
install(TARGETS _available_literal_field_family LIBRARY DESTINATION rexgraph_cpp)

set(task_outcome_sources include/task_outcome_bridge.hpp ${field_evidence_admission_sources}
  ${known_tensor_graph_sources} ${rexgraph_python_source_dir}/_task_outcome.pyx
  ${rexgraph_python_source_dir}/_task_outcome.pxd)
rexgraph_binding_receipt(task_outcome_digest ${task_outcome_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${task_outcome_sources})
set(task_outcome_output "${CMAKE_CURRENT_BINARY_DIR}/_task_outcome.cpp")
add_custom_command(OUTPUT "${task_outcome_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${task_outcome_output}"
    "${rexgraph_python_source_dir}/_task_outcome.pyx"
  DEPENDS ${task_outcome_sources} VERBATIM)
Python3_add_library(_task_outcome MODULE WITH_SOABI "${task_outcome_output}")
target_include_directories(_task_outcome PRIVATE include)
target_link_libraries(_task_outcome PRIVATE rfc::learning rfc::inference rfc::storage)
target_compile_definitions(_task_outcome PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
  RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_TASK_OUTCOME_BINDING_DIGEST="${task_outcome_digest}")
install(TARGETS _task_outcome LIBRARY DESTINATION rexgraph_cpp)

set(finite_search_sources include/finite_search_bridge.hpp include/finite_field_bridge.hpp
    include/field_evidence_bridge.hpp include/field_interface_bridge.hpp include/field_program_bridge.hpp
    ${rexgraph_python_source_dir}/_finite_search.pyx ${rexgraph_python_source_dir}/_finite_search.pxd
    ${rexgraph_python_source_dir}/_finite_field.pxd ${rexgraph_python_source_dir}/_field_interface.pxd)
rexgraph_binding_receipt(finite_search_digest ${finite_search_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${finite_search_sources})
set(finite_search_output "${CMAKE_CURRENT_BINARY_DIR}/_finite_search.cpp")
add_custom_command(OUTPUT "${finite_search_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${finite_search_output}"
    "${rexgraph_python_source_dir}/_finite_search.pyx"
  DEPENDS ${finite_search_sources} VERBATIM)
Python3_add_library(_finite_search MODULE WITH_SOABI "${finite_search_output}")
target_include_directories(_finite_search PRIVATE include)
target_link_libraries(_finite_search PRIVATE rfc::inference rfc::learning)
target_compile_features(_finite_search PRIVATE cxx_std_20)
target_compile_definitions(_finite_search PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_FINITE_SEARCH_BINDING_DIGEST="${finite_search_digest}")
install(TARGETS _finite_search LIBRARY DESTINATION rexgraph_cpp)

set(assignment_sources include/assignment_bridge.hpp ${task_outcome_sources} ${finite_field_sources}
    ${rexgraph_python_source_dir}/_assignment.pyx ${rexgraph_python_source_dir}/_assignment.pxd)
rexgraph_binding_receipt(assignment_digest ${assignment_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${assignment_sources})
set(assignment_output "${CMAKE_CURRENT_BINARY_DIR}/_assignment.cpp")
add_custom_command(OUTPUT "${assignment_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${assignment_output}"
    "${rexgraph_python_source_dir}/_assignment.pyx"
  DEPENDS ${assignment_sources} VERBATIM)
Python3_add_library(_assignment MODULE WITH_SOABI "${assignment_output}")
target_include_directories(_assignment PRIVATE include)
target_link_libraries(_assignment PRIVATE rfc::learning rfc::inference)
target_compile_features(_assignment PRIVATE cxx_std_20)
target_compile_definitions(_assignment PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_ASSIGNMENT_BINDING_DIGEST="${assignment_digest}")
install(TARGETS _assignment LIBRARY DESTINATION rexgraph_cpp)

option(REXGRAPH_BUILD_TEMPORAL_PROGRAM "Bind installed RexTemporal function and action programs" OFF)
if(REXGRAPH_BUILD_TEMPORAL_PROGRAM)
  find_package(RexTemporal 0.1.0 EXACT CONFIG REQUIRED)
  include(cmake/CompanionReceipt.cmake)
  rexgraph_companion_receipt(temporal_sdk_digest temporal_sdk_members temporal_sdk_include)
  rexgraph_install_companion_terms(rexgraph_cpp)
  if(NOT EXISTS "${temporal_sdk_include}/rex_temporal/action_programs.hpp" OR
     NOT EXISTS "${temporal_sdk_include}/rex_temporal/layer_product.hpp")
    message(FATAL_ERROR "The selected temporal SDK lacks action and layer-program contracts")
  endif()
  set(temporal_program_sources include/temporal_program_bridge.hpp
      ${rexgraph_python_source_dir}/_temporal_program.pyx
      ${rexgraph_python_source_dir}/_stalk_fields.pyx
      ${rexgraph_python_source_dir}/_field_program.pyx)
  rexgraph_binding_receipt(temporal_program_digest ${temporal_program_sources})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${temporal_program_sources} ${temporal_sdk_members})
  set(temporal_program_output "${CMAKE_CURRENT_BINARY_DIR}/_temporal_program.cpp")
  add_custom_command(OUTPUT "${temporal_program_output}"
    COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${temporal_program_output}"
      "${rexgraph_python_source_dir}/_temporal_program.pyx"
    DEPENDS ${temporal_program_sources} VERBATIM)
  Python3_add_library(_temporal_program MODULE WITH_SOABI "${temporal_program_output}")
  target_include_directories(_temporal_program PRIVATE include)
  target_link_libraries(_temporal_program PRIVATE rex_temporal::runtime)
  target_compile_features(_temporal_program PRIVATE cxx_std_20)
  target_compile_definitions(_temporal_program PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
    RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_TEMPORAL_PROGRAM_BINDING_DIGEST="${temporal_program_digest}"
    RFC_TEMPORAL_SDK_DIGEST="${temporal_sdk_digest}")
  install(TARGETS _temporal_program LIBRARY DESTINATION rexgraph_cpp)
  set(action_program_sources include/action_program_bridge.hpp ${temporal_program_sources}
      ${rexgraph_python_source_dir}/_action_program.pyx)
  rexgraph_binding_receipt(action_program_digest ${action_program_sources})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${action_program_sources})
  set(action_program_output "${CMAKE_CURRENT_BINARY_DIR}/_action_program.cpp")
  add_custom_command(OUTPUT "${action_program_output}"
    COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${action_program_output}"
      "${rexgraph_python_source_dir}/_action_program.pyx"
    DEPENDS ${action_program_sources} VERBATIM)
  Python3_add_library(_action_program MODULE WITH_SOABI "${action_program_output}")
  target_include_directories(_action_program PRIVATE include)
  target_link_libraries(_action_program PRIVATE rex_temporal::runtime)
  target_compile_features(_action_program PRIVATE cxx_std_20)
  target_compile_definitions(_action_program PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
    RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_ACTION_PROGRAM_BINDING_DIGEST="${action_program_digest}"
    RFC_TEMPORAL_SDK_DIGEST="${temporal_sdk_digest}")
  install(TARGETS _action_program LIBRARY DESTINATION rexgraph_cpp)
endif()

install(DIRECTORY "${rexgraph_python_source_dir}/" DESTINATION rexgraph_cpp
  FILES_MATCHING PATTERN "*.py" PATTERN "*.pyx" PATTERN "*.pxd" PATTERN "*.pxi")

option(REXGRAPH_BUILD_C_WORKFLOW "Bind the closed C declaration workflows" ON)
if(REXGRAPH_BUILD_C_WORKFLOW)
  if(NOT EXISTS "${rfc_include}/rfc/inference/c_declaration_archive.hpp")
    message(FATAL_ERROR "Declaration workflow binding requires the native declaration archive SDK")
  endif()
  set(c_workflow_sources include/c_workflow_bridge.hpp ${rexgraph_python_source_dir}/_c_workflow.pyx)
  rexgraph_binding_receipt(c_workflow_digest ${c_workflow_sources})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${c_workflow_sources})
  set(c_workflow_output "${CMAKE_CURRENT_BINARY_DIR}/_c_workflow.cpp")
  add_custom_command(OUTPUT "${c_workflow_output}"
    COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${c_workflow_output}"
      "${rexgraph_python_source_dir}/_c_workflow.pyx"
    DEPENDS ${c_workflow_sources} VERBATIM)
  Python3_add_library(_c_workflow MODULE WITH_SOABI "${c_workflow_output}")
  target_include_directories(_c_workflow PRIVATE include)
  target_link_libraries(_c_workflow PRIVATE rfc::inference)
  target_compile_features(_c_workflow PRIVATE cxx_std_20)
  target_compile_definitions(_c_workflow PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
    RFC_C_WORKFLOW_BINDING_DIGEST="${c_workflow_digest}")
  install(TARGETS _c_workflow LIBRARY DESTINATION rexgraph_cpp)
endif()

set(demand_sources include/demand_gluing_bridge.hpp ${rexgraph_python_source_dir}/_demand_gluing.pyx)
rexgraph_binding_receipt(demand_digest ${demand_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${demand_sources})
set(demand_output "${CMAKE_CURRENT_BINARY_DIR}/_demand_gluing.cpp")
add_custom_command(OUTPUT "${demand_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${demand_output}" "${rexgraph_python_source_dir}/_demand_gluing.pyx"
  DEPENDS ${demand_sources} VERBATIM)
Python3_add_library(_demand_gluing MODULE WITH_SOABI "${demand_output}")
target_include_directories(_demand_gluing PRIVATE include)
target_link_libraries(_demand_gluing PRIVATE rfc::inference rfc::database rfc::runtime)
target_compile_definitions(_demand_gluing PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}" RFC_DEMAND_BINDING_DIGEST="${demand_digest}")
install(TARGETS _demand_gluing LIBRARY DESTINATION rexgraph_cpp)

set(source_evolution_sources include/source_evolution_bridge.hpp
    ${rexgraph_python_source_dir}/_source_evolution.pyx ${rexgraph_python_source_dir}/_source_evolution_spec.py)
rexgraph_binding_receipt(source_evolution_digest ${source_evolution_sources})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${source_evolution_sources})
set(source_evolution_output "${CMAKE_CURRENT_BINARY_DIR}/_source_evolution.cpp")
add_custom_command(OUTPUT "${source_evolution_output}"
  COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 --output-file "${source_evolution_output}"
    "${rexgraph_python_source_dir}/_source_evolution.pyx"
  DEPENDS ${source_evolution_sources} VERBATIM)
Python3_add_library(_source_evolution MODULE WITH_SOABI "${source_evolution_output}")
target_include_directories(_source_evolution PRIVATE include)
target_link_libraries(_source_evolution PRIVATE rfc::storage)
target_compile_features(_source_evolution PRIVATE cxx_std_20)
target_compile_definitions(_source_evolution PRIVATE RFC_ENGINE_DIGEST="${rfc_digest}"
    RFC_ENGINE_VERSION="${RFC_VERSION}" RFC_SOURCE_EVOLUTION_BINDING_DIGEST="${source_evolution_digest}")
install(TARGETS _source_evolution LIBRARY DESTINATION rexgraph_cpp)
