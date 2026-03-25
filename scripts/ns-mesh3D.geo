// Parameterized 3D flow-past-cylinder mesh.
//
// Example generation commands:
//   gmsh scripts/ns-mesh3D.geo -3 -format msh4 -setnumber level 0 -o mesh/navierstokes3D_L0.msh
//   gmsh scripts/ns-mesh3D.geo -3 -format msh4 -setnumber level 1 -o mesh/navierstokes3D_L1.msh
//   gmsh scripts/ns-mesh3D.geo -3 -format msh4 -setnumber level 2 -o mesh/navierstokes3D_L2.msh
//   gmsh scripts/ns-mesh3D.geo -3 -format msh4 -setnumber level 3 -o mesh/navierstokes3D_L3.msh

SetFactory("OpenCASCADE");

DefineConstant[
  level = {0, Choices{0, 1, 2, 3}, Name "Mesh/Level"},
  base_lc_coarse = {1.00, Min 0.01, Max 5.0, Step 0.01, Name "Mesh/Base coarse size"},
  base_lc_fine = {0.05, Min 0.001, Max 1.0, Step 0.001, Name "Mesh/Base cylinder size"},
  wake_size_factor = {2.0, Min 1.0, Max 10.0, Step 0.1, Name "Mesh/Wake size factor"},
  cylinder_band = {1.20, Min 0.05, Max 5.0, Step 0.05, Name "Mesh/Cylinder refinement band"},
  wake_length = {8.0, Min 0.5, Max 20.0, Step 0.5, Name "Mesh/Wake length"},
  wake_half_height = {0.80, Min 0.1, Max 4.0, Step 0.05, Name "Mesh/Wake half height"},
  span = {1.0, Min 0.1, Max 10.0, Step 0.1, Name "Geometry/Spanwise length"}
];

// Halving h at each level gives roughly 8x more cells/DoFs in 3D.
If(level == 0)
  scale = 1.0;
ElseIf(level == 1)
  scale = 0.5;
ElseIf(level == 2)
  scale = 0.25;
Else
  scale = 0.125;
EndIf

lc_coarse = base_lc_coarse * scale;
lc_fine = base_lc_fine * scale;
lc_wake = wake_size_factor * lc_fine;

// Geometry parameters.
x_min = 0.0;
x_max = 25.0;
y_min = 0.0;
y_max = 4.1;
z_min = 0.0;
z_max = span;

x_c = 2.0;
y_c = 2.0;
radius = 0.5;

// Slightly extend the cylinder so the subtraction cleanly pierces both z faces.
eps = 1e-3;

Box(1) = {x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min};
Cylinder(2) = {x_c, y_c, z_min - eps, 0, 0, (z_max - z_min) + 2 * eps, radius};

fluid[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Use tighter CAD bounds so these selections stay stable after the boolean cut.
Geometry.OCCBoundsUseStl = 1;

inlet[]    = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_min + eps, y_max + eps, z_max + eps};
outlet[]   = Surface In BoundingBox{x_max - eps, y_min - eps, z_min - eps, x_max + eps, y_max + eps, z_max + eps};
bottom[]   = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_max + eps, y_min + eps, z_max + eps};
top[]      = Surface In BoundingBox{x_min - eps, y_max - eps, z_min - eps, x_max + eps, y_max + eps, z_max + eps};
front[]    = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_max + eps, y_max + eps, z_min + eps};
back[]     = Surface In BoundingBox{x_min - eps, y_min - eps, z_max - eps, x_max + eps, y_max + eps, z_max + eps};
cylinder[] = Surface In BoundingBox{x_c - radius - eps, y_c - radius - eps, z_min - eps, x_c + radius + eps, y_c + radius + eps, z_max + eps};

// Refinement boxes in 3D: one around the cylinder, one in the wake.
Field[1] = Box;
Field[1].VIn = lc_fine;
Field[1].VOut = lc_coarse;
Field[1].XMin = x_c - radius - cylinder_band;
Field[1].XMax = x_c + radius + cylinder_band;
Field[1].YMin = y_c - radius - cylinder_band;
Field[1].YMax = y_c + radius + cylinder_band;
Field[1].ZMin = z_min;
Field[1].ZMax = z_max;

Field[2] = Box;
Field[2].VIn = lc_wake;
Field[2].VOut = lc_coarse;
Field[2].XMin = x_c + radius;
Field[2].XMax = x_c + radius + wake_length;
Field[2].YMin = y_c - wake_half_height;
Field[2].YMax = y_c + wake_half_height;
Field[2].ZMin = z_min;
Field[2].ZMax = z_max;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Physical Surface("Inlet", 1) = {inlet[0]};
Physical Surface("Outlet", 2) = {outlet[0]};
// The z-min/z-max faces are grouped with the channel walls for a simple no-slip setup.
Physical Surface("Walls", 3) = {bottom[0], top[0], front[0], back[0]};
Physical Surface("Cylinder", 5) = {cylinder[0]};
Physical Volume("Fluid", 11) = {fluid[0]};

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
