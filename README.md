# Mesh Simplification — COMP557 A3

**Yantian Yin — 261143026**

An interactive 3D mesh simplification tool using the **Half-Edge Data Structure (HEDS)** and **Quadric Error Metrics (QEM)**. Supports progressive LOD reduction on standard mesh files (bunny, cow, teapot, etc.) with a real-time OpenGL viewer.

> **Skills:** Python · OpenGL (ModernGL) · PyQt5 · Half-edge data structure · Quadric error metrics · Mesh geometry · GLSL shaders · Linear algebra (pseudo-inverse)

---

## What I Built

### Q1 — Half-Edge Data Structure
Built a full HEDS from scratch supporting:
- `HalfEdge`, `Vertex`, `Face` classes with `head`, `twin`, `next`, `face` pointers
- Ring-1 neighbour traversal (used extensively for collapse and quadric computation)
- Face normals and centroids computed on demand

### Q2 & Q3 — Edge Collapse
Implemented edge collapse to merge two vertices `a` and `b` into a new vertex:
1. **Pre-collapse**: traverse ring-1 of `(a, b)` and collect all inward half-edges and old face geometry
2. **Collapse**: redirect all inward half-edges of `a` and `b` to the new merged vertex; re-twin the two pairs of boundary half-edges; swap deleted faces to the end of the face list
3. **Post-collapse**: collect all neighbours of the new vertex as `affected_faces`; recompute `new_faces` geometry by traversing the new vertex's half-edge ring

### Q4 — Link Condition (Topology Safety Check)
Before collapsing an edge, checks that `a` and `b` share **at most 2** common neighbours — the link condition that prevents non-manifold topology from being created.

### Q5 — Quadric Error Metric
For each candidate edge collapse:
- Computes per-vertex quadric `Q = Σ Kᵢ` where `Kᵢ = ppᵀ` for each neighbouring face plane
- Solves `Av = -b` for the optimal new vertex position
- Handles rank-deficient systems via **pseudo-inverse**
- Falls back to endpoints and midpoint as candidates; picks lowest-cost position
- Stores optimal position and cost in `EdgeCollapseData`

### Q6 — Sorted Edge List Maintenance
After each collapse:
- Removes `EdgeCollapseData` of all affected edges from the sorted priority list
- Reinitialises `EdgeCollapseData` for all new edges and re-inserts them
- Enables greedy LOD simplification by always collapsing the lowest-cost edge next

---

## Usage

```bash
pip install moderngl PyQt5 pyglm trimesh sortedcontainers

python a3_app.py
```

Load a mesh from the `data/` folder and use the LOD slider to simplify interactively.

---

## Supported Meshes

| Mesh | Description |
|------|-------------|
| `icoSphere.obj` | Simple sphere — good for testing topology |
| `cube.obj` / `cube2.obj` | Box primitives |
| `tetrahedron.obj` | Minimal manifold mesh |
| `bunny.obj` / `bunny-BIG.obj` | Stanford bunny |
| `cow.obj` | Standard test mesh |
| `topologyTest.obj` | Edge-case topology for link condition testing |

---

## Project Structure

| File | Description |
|------|-------------|
| `heds.py` | Half-edge data structure: `HalfEdge`, `Vertex`, `Face`, `EdgeCollapseData`, `CollapseRecord` |
| `simplification_viewer.py` | OpenGL viewer with LOD rendering, wireframe, half-edge and ID overlays |
| `a3_app.py` | PyQt5 app entry point |
| `controls.py` | UI control panel (LOD slider, mesh loader, debug toggles) |
| `glsl/` | GLSL shaders for mesh and half-edge line rendering |
| `data/` | OBJ mesh files |
