add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC NeoMathEngine NeoML dnnl)

target_include_directories(${PROJECT_NAME} PUBLIC
    ../third_party/NeoML/NeoML/include
    ../third_party/oneDNN/include)

install(
    TARGETS ${PROJECT_NAME} NeoML NeoMathEngine dnnl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_BINDIR})
