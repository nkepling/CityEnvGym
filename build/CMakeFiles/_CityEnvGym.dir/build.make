# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build

# Include any dependencies generated for this target.
include CMakeFiles/_CityEnvGym.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/_CityEnvGym.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/_CityEnvGym.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/_CityEnvGym.dir/flags.make

CMakeFiles/_CityEnvGym.dir/codegen:
.PHONY : CMakeFiles/_CityEnvGym.dir/codegen

CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o: CMakeFiles/_CityEnvGym.dir/flags.make
CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o: /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/src/binder.cpp
CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o: CMakeFiles/_CityEnvGym.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o -MF CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o.d -o CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o -c /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/src/binder.cpp

CMakeFiles/_CityEnvGym.dir/src/binder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/_CityEnvGym.dir/src/binder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/src/binder.cpp > CMakeFiles/_CityEnvGym.dir/src/binder.cpp.i

CMakeFiles/_CityEnvGym.dir/src/binder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/_CityEnvGym.dir/src/binder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/src/binder.cpp -o CMakeFiles/_CityEnvGym.dir/src/binder.cpp.s

# Object files for target _CityEnvGym
_CityEnvGym_OBJECTS = \
"CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o"

# External object files for target _CityEnvGym
_CityEnvGym_EXTERNAL_OBJECTS =

_CityEnvGym.cpython-310-darwin.so: CMakeFiles/_CityEnvGym.dir/src/binder.cpp.o
_CityEnvGym.cpython-310-darwin.so: CMakeFiles/_CityEnvGym.dir/build.make
_CityEnvGym.cpython-310-darwin.so: libCityEnv.a
_CityEnvGym.cpython-310-darwin.so: CMakeFiles/_CityEnvGym.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module _CityEnvGym.cpython-310-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_CityEnvGym.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip -x /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build/_CityEnvGym.cpython-310-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/_CityEnvGym.dir/build: _CityEnvGym.cpython-310-darwin.so
.PHONY : CMakeFiles/_CityEnvGym.dir/build

CMakeFiles/_CityEnvGym.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/_CityEnvGym.dir/cmake_clean.cmake
.PHONY : CMakeFiles/_CityEnvGym.dir/clean

CMakeFiles/_CityEnvGym.dir/depend:
	cd /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build /Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/CityEnvGym/build/CMakeFiles/_CityEnvGym.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/_CityEnvGym.dir/depend

