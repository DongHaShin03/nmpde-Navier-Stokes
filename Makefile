BUILD_DIR ?= build
MESH_DIR ?= mesh
GMSH ?= gmsh
TOOLS_DIR ?= tools
GMSH_VERSION ?= 4.15.0
GMSH_ARCHIVE := gmsh-$(GMSH_VERSION)-Linux64.tgz
GMSH_URL := https://gmsh.info/bin/Linux/$(GMSH_ARCHIVE)
GMSH_ROOT := $(TOOLS_DIR)/gmsh-$(GMSH_VERSION)-Linux64
GMSH_LOCAL := $(GMSH_ROOT)/bin/gmsh
MESH_LEVELS ?= 0 1
MESH_2D_FORMAT ?= msh4
MESH_3D_FORMAT ?= msh2
MESH_STAMP := $(MESH_DIR)/.dir
MESH_2D_FILES := $(addprefix $(MESH_DIR)/navierstokes_L,$(addsuffix .msh,$(MESH_LEVELS)))
MESH_3D_FILES := $(addprefix $(MESH_DIR)/navierstokes3D_L,$(addsuffix .msh,$(MESH_LEVELS)))

.PHONY: all configure clean clear gmsh gmsh-local meshes meshes-local-gmsh mesh mesh-2d mesh-3d mesh-2d-local-gmsh mesh-3d-local-gmsh 2D 3D

all: configure
	cmake --build "$(BUILD_DIR)"

configure:
	cmake -S . -B "$(BUILD_DIR)"

clean:
	@test -n "$(BUILD_DIR)"
	@test "$(abspath $(BUILD_DIR))" != "$(abspath .)"
	rm -rf -- "$(BUILD_DIR)"

clear: clean

gmsh: $(GMSH_LOCAL)

gmsh-local: gmsh

$(GMSH_LOCAL):
	mkdir -p "$(TOOLS_DIR)"
	if command -v wget >/dev/null 2>&1; then \
		wget -O "$(TOOLS_DIR)/$(GMSH_ARCHIVE)" "$(GMSH_URL)"; \
	else \
		curl -L -o "$(TOOLS_DIR)/$(GMSH_ARCHIVE)" "$(GMSH_URL)"; \
	fi
	tar -xzf "$(TOOLS_DIR)/$(GMSH_ARCHIVE)" -C "$(TOOLS_DIR)"
	test -x "$(GMSH_LOCAL)"

meshes: mesh-2d mesh-3d

meshes-local-gmsh: gmsh
	$(MAKE) meshes MESH_LEVELS="$(MESH_LEVELS)" MESH_2D_FORMAT="$(MESH_2D_FORMAT)" MESH_3D_FORMAT="$(MESH_3D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH_LOCAL)"

mesh:
	@if printf '%s\n' "$(MAKECMDGOALS)" | grep -qw '2D'; then \
		$(MAKE) mesh-2d MESH_LEVELS="$(MESH_LEVELS)" MESH_2D_FORMAT="$(MESH_2D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH)"; \
	elif printf '%s\n' "$(MAKECMDGOALS)" | grep -qw '3D'; then \
		$(MAKE) mesh-3d MESH_LEVELS="$(MESH_LEVELS)" MESH_3D_FORMAT="$(MESH_3D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH)"; \
	else \
		$(MAKE) meshes MESH_LEVELS="$(MESH_LEVELS)" MESH_2D_FORMAT="$(MESH_2D_FORMAT)" MESH_3D_FORMAT="$(MESH_3D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH)"; \
	fi

ifneq ($(filter mesh,$(MAKECMDGOALS)),)
2D:
	@:

3D:
	@:
else
2D: mesh-2d

3D: mesh-3d
endif

mesh-2d: $(MESH_2D_FILES)

mesh-3d: $(MESH_3D_FILES)

mesh-2d-local-gmsh: gmsh
	$(MAKE) mesh-2d MESH_LEVELS="$(MESH_LEVELS)" MESH_2D_FORMAT="$(MESH_2D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH_LOCAL)"

mesh-3d-local-gmsh: gmsh
	$(MAKE) mesh-3d MESH_LEVELS="$(MESH_LEVELS)" MESH_3D_FORMAT="$(MESH_3D_FORMAT)" MESH_DIR="$(MESH_DIR)" GMSH="$(GMSH_LOCAL)"

$(MESH_STAMP):
	mkdir -p "$(MESH_DIR)"
	touch "$@"

$(MESH_DIR)/navierstokes_L%.msh: scripts/flow_past_cylinder_2d.geo | $(MESH_STAMP)
	$(GMSH) "$<" -2 -format "$(MESH_2D_FORMAT)" -setnumber level "$*" -o "$@"

$(MESH_DIR)/navierstokes3D_L%.msh: scripts/flow_past_cylinder_3d.geo | $(MESH_STAMP)
	$(GMSH) "$<" -3 -format "$(MESH_3D_FORMAT)" -setnumber level "$*" -o "$@"
