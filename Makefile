BUILD_DIR ?= build

.PHONY: all configure clean

all: configure
	cmake --build "$(BUILD_DIR)"

configure:
	cmake -S . -B "$(BUILD_DIR)"

clean:
	@test -n "$(BUILD_DIR)"
	@test "$(abspath $(BUILD_DIR))" != "$(abspath .)"
	rm -rf -- "$(BUILD_DIR)"
