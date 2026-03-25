// Parameterized flow-past-cylinder mesh derived from navierstokesFINAL.geo
//
// Example generation commands:
//   gmsh mesh/navierstokesFINAL_levels.geo -2 -format msh4 -setnumber level 0 -o mesh/navierstokesFINAL_L0.msh
//   gmsh mesh/navierstokesFINAL_levels.geo -2 -format msh4 -setnumber level 1 -o mesh/navierstokesFINAL_L1.msh
//   gmsh mesh/navierstokesFINAL_levels.geo -2 -format msh4 -setnumber level 2 -o mesh/navierstokesFINAL_L2.msh
//   gmsh mesh/navierstokesFINAL_levels.geo -2 -format msh4 -setnumber level 3 -o mesh/navierstokesFINAL_L3.msh
//
// If your executables still expect ../mesh/navierstokesFINAL.msh, just change the
// output name in the commands above to mesh/navierstokesFINAL.msh.

SetFactory("OpenCASCADE");

DefineConstant[
  level = {0, Choices{0, 1, 2, 3}, Name "Mesh/Level"},
  base_lc_coarse = {1.00, Min 0.01, Max 5.0, Step 0.01, Name "Mesh/Base coarse size"},
  base_lc_fine = {0.05, Min 0.001, Max 1.0, Step 0.001, Name "Mesh/Base cylinder size"},
  wake_size_factor = {2.0, Min 1.0, Max 10.0, Step 0.1, Name "Mesh/Wake size factor"},
  cylinder_band = {1.20, Min 0.05, Max 5.0, Step 0.05, Name "Mesh/Cylinder refinement band"},
  wake_length = {8.0, Min 0.5, Max 20.0, Step 0.5, Name "Mesh/Wake length"},
  wake_half_height = {0.80, Min 0.1, Max 4.0, Step 0.05, Name "Mesh/Wake half height"}
];

// Halving h at each level gives roughly 4x more cells/DoFs in 2D.
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

x_c = 2.0;
y_c = 2.0;
radius = 0.5;

// Outer rectangle.
Point(1) = {x_min, y_min, 0, 1.0};
Point(2) = {x_max, y_min, 0, 1.0};
Point(3) = {x_min, y_max, 0, 1.0};
Point(4) = {x_max, y_max, 0, 1.0};

Line(1) = {3, 1}; // Inlet
Line(2) = {1, 2}; // Bottom wall
Line(3) = {2, 4}; // Outlet
Line(4) = {4, 3}; // Top wall

// Cylinder.
Circle(5) = {x_c, y_c, 0, radius, 0, 2 * Pi};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

// Refinement field around the cylinder.
Field[1] = Distance;
Field[1].CurvesList = {5};
Field[1].Sampling = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_fine;
Field[2].SizeMax = lc_coarse;
Field[2].DistMin = 0.0;
Field[2].DistMax = cylinder_band;

// Extra refinement in the wake so all levels preserve the same flow features.
Field[3] = Box;
Field[3].VIn = lc_wake;
Field[3].VOut = lc_coarse;
Field[3].XMin = x_c + radius;
Field[3].XMax = x_c + radius + wake_length;
Field[3].YMin = y_c - wake_half_height;
Field[3].YMax = y_c + wake_half_height;

Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

// Let the background field drive the sizing almost entirely.
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Physical Curve("Inlet", 1) = {1};
Physical Curve("Outlet", 2) = {3};
Physical Curve("Walls", 3) = {2, 4};
Physical Curve("Cylinder", 5) = {5};
Physical Surface("Fluid", 11) = {1};

// Frontal-Delaunay is usually a good default in 2D.
Mesh.Algorithm = 6;