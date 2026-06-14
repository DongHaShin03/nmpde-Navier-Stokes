SetFactory("OpenCASCADE");

DefineConstant[
  level = {0, Choices{0, 1, 2, 3}, Name "Mesh/Level"},

  h_far_base      = {0.15,  Min 0.005,  Max 0.50,  Step 0.001,  Name "Mesh/Far-field size"},
  h_wall_base     = {0.08,  Min 0.002,  Max 0.20,  Step 0.001,  Name "Mesh/Wall-band size"},
  h_wake_base     = {0.035,  Min 0.001,  Max 0.10,  Step 0.001,  Name "Mesh/Wake size"},
  h_cylinder_base = {0.015, Min 0.0005, Max 0.05,  Step 0.0005, Name "Mesh/Cylinder size"},

  cylinder_band     = {0.12,  Min 0.02, Max 0.50,  Step 0.01,  Name "Mesh/Cylinder refinement band"},
  cylinder_core     = {0.02,  Min 0.00, Max 0.10,  Step 0.005, Name "Mesh/Cylinder finest core"},
  wake_length       = {1.75,  Min 0.20, Max 1.95,  Step 0.05,  Name "Mesh/Wake length"},
  wake_half_height  = {0.12,  Min 0.03, Max 0.205, Step 0.005, Name "Mesh/Wake half height"},
  wall_band         = {0.035, Min 0.00, Max 0.15,  Step 0.005, Name "Mesh/Wall refinement band"}
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
x_max = 2.5;
y_min = 0.0;
y_max = 0.41;
z_min = 0.0;
z_max = 0.41;

x_c = 0.50;
y_c = 0.20;
radius = 0.05;

H = y_max - y_min;
D = 2.0 * radius;

eps = 1e-6;

Box(1) = {x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min};
Cylinder(2) = {x_c, y_c, z_min - eps, 0, 0, (z_max - z_min) + 2 * eps, radius};

fluid[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

Geometry.OCCBoundsUseStl = 1;

inlet[]    = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_min + eps, y_max + eps, z_max + eps};
outlet[]   = Surface In BoundingBox{x_max - eps, y_min - eps, z_min - eps, x_max + eps, y_max + eps, z_max + eps};
bottom[]   = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_max + eps, y_min + eps, z_max + eps};
top[]      = Surface In BoundingBox{x_min - eps, y_max - eps, z_min - eps, x_max + eps, y_max + eps, z_max + eps};
front[]    = Surface In BoundingBox{x_min - eps, y_min - eps, z_min - eps, x_max + eps, y_max + eps, z_min + eps};
back[]     = Surface In BoundingBox{x_min - eps, y_min - eps, z_max - eps, x_max + eps, y_max + eps, z_max + eps};
cylinder[] = Surface In BoundingBox{x_c - radius - eps, y_c - radius - eps, z_min - eps, x_c + radius + eps, y_c + radius + eps, z_max + eps};

Field[1] = Distance;
Field[1].SurfacesList = {cylinder[]};
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
Field[3].ZMin = z_min;
Field[3].ZMax = z_max;

Field[4] = Box;
Field[4].VIn = h_wall;
Field[4].VOut = h_far;
Field[4].XMin = x_min;
Field[4].XMax = x_max;
Field[4].YMin = y_min;
Field[4].YMax = y_min + wall_band;
Field[4].ZMin = z_min;
Field[4].ZMax = z_max;

Field[5] = Box;
Field[5].VIn = h_wall;
Field[5].VOut = h_far;
Field[5].XMin = x_min;
Field[5].XMax = x_max;
Field[5].YMin = y_max - wall_band;
Field[5].YMax = y_max;
Field[5].ZMin = z_min;
Field[5].ZMax = z_max;

Field[6] = Box;
Field[6].VIn = h_wall;
Field[6].VOut = h_far;
Field[6].XMin = x_min;
Field[6].XMax = x_max;
Field[6].YMin = y_min;
Field[6].YMax = y_max;
Field[6].ZMin = z_min;
Field[6].ZMax = z_min + wall_band;

Field[7] = Box;
Field[7].VIn = h_wall;
Field[7].VOut = h_far;
Field[7].XMin = x_min;
Field[7].XMax = x_max;
Field[7].YMin = y_min;
Field[7].YMax = y_max;
Field[7].ZMin = z_max - wall_band;
Field[7].ZMax = z_max;

Field[8] = Min;
Field[8].FieldsList = {2, 3, 4, 5, 6, 7};
Background Field = 8;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.RecombineAll = 0;
Mesh.Algorithm = 6;        
Mesh.Algorithm3D = 1;     
Mesh.Smoothing = 10;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Physical Surface("Inlet", 1) = {inlet[]};
Physical Surface("Outlet", 2) = {outlet[]};
Physical Surface("Walls", 3) = {bottom[], top[], front[], back[]};
Physical Surface("Cylinder", 5) = {cylinder[]};
Physical Volume("Fluid", 11) = {fluid[]};
