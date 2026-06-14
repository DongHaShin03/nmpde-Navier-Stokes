SetFactory("OpenCASCADE");

DefineConstant[
  level = {0, Choices{0, 1, 2, 3}, Name "Mesh/Level"},
  h_far_base      = {0.055,  Min 0.005,  Max 0.50,  Step 0.001, Name "Mesh/Far-field size"},
  h_wall_base     = {0.025,  Min 0.002,  Max 0.20,  Step 0.001, Name "Mesh/Wall-band size"},
  h_wake_base     = {0.012,  Min 0.001,  Max 0.10,  Step 0.001, Name "Mesh/Wake size"},
  h_cylinder_base = {0.0045, Min 0.0005, Max 0.05,  Step 0.0005, Name "Mesh/Cylinder size"},

  //Refinement Region
  cylinder_band     = {0.12, Min 0.02, Max 0.50, Step 0.01, Name "Mesh/Cylinder refinement band"},
  cylinder_core     = {0.02, Min 0.00, Max 0.10, Step 0.005, Name "Mesh/Cylinder finest core"},
  wake_length       = {1.75, Min 0.20, Max 2.00, Step 0.05, Name "Mesh/Wake length"},
  wake_half_height  = {0.12, Min 0.03, Max 0.205, Step 0.005, Name "Mesh/Wake half height"},
  wall_band         = {0.035, Min 0.00, Max 0.15, Step 0.005, Name "Mesh/Wall refinement band"}
];

If(level == 0)
  scale = 1.0;
ElseIf(level == 1)
  scale = 0.5;
ElseIf(level == 2)
  scale = 0.25;
Else
  scale = 0.125;
EndIf

h_far      = h_far_base      * scale;
h_wall     = h_wall_base     * scale;
h_wake     = h_wake_base     * scale;
h_cylinder = h_cylinder_base * scale;

x_min = 0.0;
x_max = 2.2;
y_min = 0.0;
y_max = 0.41;

x_c = 0.20;
y_c = 0.20;
radius = 0.05;

Point(1) = {x_min, y_min, 0, h_far};
Point(2) = {x_max, y_min, 0, h_far};
Point(3) = {x_min, y_max, 0, h_far};
Point(4) = {x_max, y_max, 0, h_far};

Line(1) = {3, 1}; // Inlet
Line(2) = {1, 2}; // Bottom wall
Line(3) = {2, 4}; // Outlet
Line(4) = {4, 3}; // Top wall

Circle(5) = {x_c, y_c, 0, radius, 0, 2 * Pi};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

Field[1] = Distance;
Field[1].CurvesList = {5};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_cylinder;
Field[2].SizeMax = h_far;
Field[2].DistMin = cylinder_core;
Field[2].DistMax = cylinder_band;

Field[3] = Box;
Field[3].VIn = h_wake;
Field[3].VOut = h_far;
Field[3].XMin = x_c + radius;
Field[3].XMax = x_c + radius + wake_length;
Field[3].YMin = y_c - wake_half_height;
Field[3].YMax = y_c + wake_half_height;

Field[4] = Box;
Field[4].VIn = h_wall;
Field[4].VOut = h_far;
Field[4].XMin = x_min;
Field[4].XMax = x_max;
Field[4].YMin = y_min;
Field[4].YMax = y_min + wall_band;

Field[5] = Box;
Field[5].VIn = h_wall;
Field[5].VOut = h_far;
Field[5].XMin = x_min;
Field[5].XMax = x_max;
Field[5].YMin = y_max - wall_band;
Field[5].YMax = y_max;

Field[6] = Min;
Field[6].FieldsList = {2, 3, 4, 5};
Background Field = 6;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.Algorithm = 6;     
Mesh.Smoothing = 10;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Physical Curve("Inlet", 1) = {1};
Physical Curve("Outlet", 2) = {3};
Physical Curve("Walls", 3) = {2, 4};
Physical Curve("Cylinder", 5) = {5};
Physical Surface("Fluid", 11) = {1};
