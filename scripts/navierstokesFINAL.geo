// --- INIZIO SCRIPT ---
SetFactory("OpenCASCADE");

// 1. DEFINIZIONE DELLE GRANDEZZE DELLA MESH
// Puoi cambiare questi numeri per rendere la mesh più o meno fitta
lc_coarse = 1.0;  // Dimensione mesh lontano dal cilindro (Inlet/Outlet)
lc_fine   = 0.05;  // Dimensione mesh SUL cilindro (molto più fitta)

// 2. GEOMETRIA
// Assegno lc_coarse ai punti del rettangolo esterno
Point(1) = {0, 0, 0, lc_coarse};
Point(2) = {25, 0, 0, lc_coarse};
Point(3) = {0, 4.1, 0, lc_coarse};
Point(4) = {25, 4.1, 0, lc_coarse};

Line(1) = {3, 1}; // Inlet
Line(2) = {1, 2}; // Bottom
Line(3) = {2, 4}; // Outlet
Line(4) = {4, 3}; // Top

// Il Cilindro
// Nota: Riduco il raggio a 0.5 per evitare che tocchi le pareti se usi parametri standard,
// ma se vuoi il raggio originale cambia 0.5 con 1.0.
Circle(5) = {2, 2, 0, 0.5, 0, 2*Pi};

// 3. RAFFINAMENTO LOCALE (Il trucco per la mesh fitta)
// Questo comando dice: "Prendi tutti i punti che appartengono alla Curva 5 (il cerchio)
// e imposta la loro dimensione mesh a lc_fine"
MeshSize { PointsOf{ Curve{5}; } } = lc_fine;

// 4. CREAZIONE SUPERFICIE
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

// 5. GRUPPI FISICI (IDs corretti dalla discussione precedente)
Physical Curve("Inlet", 1) = {1};
Physical Curve("Outlet", 2) = {3};
Physical Curve("Walls", 3) = {2, 4};
Physical Curve("Cylinder", 5) = {5};
Physical Surface("Fluid", 11) = {1};

// 6. ALGORITMO DI MESHING
// Frontal-Delaunay produce solitamente transizioni più belle in 2D
Mesh.Algorithm = 6;