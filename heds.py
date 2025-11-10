import numpy as np
from pyglm import glm


class HalfEdge:

    def __init__(self, head: 'Vertex', face: 'Face', twin: 'HalfEdge'):
        self.head = head  # the vertex at the "head" of this half-edge
        self.face = face  # left face that this half-edge borders
        self.twin = twin  # the twin half-edge (None if boundary)
        self.next = None  # the next half-edge in the face (to be set later)
        self.edge_collapse_data = None  # data for edge collapse operation, to be set later
        if head.he is None:
            head.he = self  # set the vertex's half-edge if not already set
        if face.he is None:
            face.he = self  # set the face's half-edge if not already set

    def tail(self):
        """ Get the tail of this half-edge."""
        he = self
        while he.next is not self:
            he = he.next  # previous half edge
        return he.head

    def __str__(self):
        return f"~~ HE with Head {self.head.index}, Face {self.face.index} ~~"


class Face:

    def __init__(self, index: int, he: HalfEdge = None):
        """ A face in the half-edge data structure. """
        self.index = index
        self.he = he  # one of the half-edges bordering this face
        self.normal = None  # normal of this face, for visualization and otherwise only for inital quadric computation
        self.center = None  # center of this face, for visualization
        self.M = None  # model matrix for text rendering
        self.text_scale = None  # scale for text rendering

    def get_normal(self):
        """ Return this face's normal. Will compute when called for the first time on a face."""
        if self.normal is not None:
            return self.normal
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        n = glm.normalize(glm.cross(glm.vec3(*(v1 - v0)), glm.vec3(*(v2 - v0))))
        self.normal = n
        return n

    def get_center(self):
        """ Return this face's centroid. Will compute when called for the first time on a face."""
        if self.center is not None:
            return self.center
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        c = (v0 + v1 + v2) / 3.0
        self.center = c
        return c

    def draw_debug(self, P: glm.mat4, V: glm.mat4, faces: np.ndarray, vert_objs: list, text_renderer):
        """Render the index of this face, for debug purposes."""
        if self.M is None:
            # Cache the necessary quantities, with redo/undo of collapses causing cache recompute by setting M to None
            # We're using the np faces array to get the vertex indices for this face
            #   because the half-edge structure may have changed
            v0 = vert_objs[faces[self.index, 0]].pos
            v1 = vert_objs[faces[self.index, 1]].pos
            v2 = vert_objs[faces[self.index, 2]].pos
            ave_edge_length = (glm.length(v0 - v1) + glm.length(v1 - v2) + glm.length(v2 - v0)) / 3.0
            center = (v0 + v1 + v2) / 3.0
            n = glm.cross(v1 - v0, v2 - v0)
            if glm.length(n) < 1e-6:
                n = glm.vec3(0, 0, 1)
            else:
                n = glm.normalize(n)
            t = glm.normalize(v1 - v0)
            b = glm.normalize(glm.cross(n, t))
            self.M = glm.mat4(
                glm.vec4(t, 0.0),  # X axis
                glm.vec4(b, 0.0),  # Y axis
                glm.vec4(n, 0.0),  # Z axis
                glm.vec4(center + n * 0.01, 1.0)  # Translation
            )
            self.text_scale = ave_edge_length * 0.1
        text_renderer.render_text(str(self.index), P, V * self.M, color=glm.vec4(1, 1, 1, 1),
                                  char_width=self.text_scale, centered=True, view_aligned=False)

    def __str__(self):
        return f"~~ Face with Index {self.index}, Referencing HE {self.he} ~~"


def compute_face_quadric(face: Face) -> glm.mat4:
    """Compute and return the quadric error matrix K = pp^T for the given triangular face.
    The plane equation is p = [a, b, c, d]^T where ax + by + cz + d = 0.
    For a plane with normal n and point v0: n·v + d = 0, so d = -n·v0."""
    he0 = face.he
    v0 = he0.head.pos

    n = face.get_normal()
    n = glm.normalize(n)
    d = -glm.dot(n, v0)
    # Plane vector: p = [nx, ny, nz, d]^T
    plane = glm.vec4(n.x, n.y, n.z, d)
    # Quadric: K = pp^T (outer product)
    return glm.outerProduct(plane, plane)


def mat4_to_numpy(mat: glm.mat4) -> np.ndarray:
    """Convert a glm.mat4 into a 4x4 numpy array (row-major)."""
    return np.array([[float(mat[c][r]) for c in range(4)] for r in range(4)], dtype=float)


class Vertex:

    def __init__(self, index: int, pos: np.ndarray, he: HalfEdge):
        """ A vertex in the half-edge data structure
        Args:
            index: index of this vertex in the vertex list
            pos: 3D position of this vertex (np for convenience, as this is coming from trimesh)
            he: one of the half-edges ending at this vertex
        """
        self.index = index
        self.pos = glm.vec3(*pos)  # 3D position of this vertex
        self.Q = glm.mat4(1)  # Quadric
        self.he = he  # one of the half-edges ending at this vertex
        self.normal = None  # average normal of faces around this vertex, for visualization
        self.removed_at_level = None  # level of detail at which this vertex was removed
        self.cost = 0  # cost of this vertex living where it is

        self.text_pos = None  # Data for debug text
        self.text_scale = None
    def compute_Q(self):
        """ Compute the quadric Q for this vertex by summing quadrics K from all adjacent faces.
        Q = Σ K_i where K_i is the quadric of face i adjacent to this vertex.
        It gets stored in the parameter self.Q"""

        self.Q = glm.mat4(0)

        if self.he is None:
            return

        # Traverse around the vertex: start with a half-edge that has this vertex as head
        # Then move to the next face via next.twin
        start = self.he
        current = start
        while True:
            self.Q += compute_face_quadric(current.face)
            current = current.next.twin
            if current == start:
                break


    def get_normal(self) -> glm.vec3:
        """ Compute the average normal of faces adjacent to this vertex.
        The value is cached after first computation.
        This is currently only used for visualization, but could also be
        used for smooth shading of the mesh."""
        if self.normal is not None:
            return self.normal
        n = glm.vec3(0, 0, 0)
        h = self.he
        while True:
            # Accumulate normal value
            n += h.face.get_normal()
            h = h.next.twin
            if h == self.he:
                break
        if glm.length(n) > 1e-6:
            n = glm.normalize(n)
        self.normal = n
        return n

    def compute_debug_viz_data(self):
        """ Compute data for debug visualization (text position and scale)
        Note that this should be called when the vertex is first created so
        that it has access to a valid half-edge structure around it."""
        # Use the average edge length around this vertex to scale the text
        edge_length = 0.0
        num_edges = 0
        h = self.he
        while True:
            tail = h.tail().pos
            edge_length += glm.length(self.pos - tail)
            num_edges += 1
            h = h.next.twin
            if h == self.he:
                break
        avg_edge_length = edge_length / num_edges if num_edges > 0 else 0.0
        self.text_scale = avg_edge_length * 0.1
        self.text_pos = self.pos + self.get_normal() * avg_edge_length * 0.1

    def draw_debug(self, P: glm.mat4, V: glm.mat4, text_renderer):
        """Render the index of this vertex, for debug purposes."""
        text_renderer.render_text(str(self.index), P, V, pos=self.text_pos, char_width=self.text_scale,
                                  color=glm.vec4(0, 0.75, 0, 1), centered=True, view_aligned=True)

    def __str__(self):
        return f"~~ Vertex with Index {self.index}, Referencing HE {self.he} ~~"


class EdgeCollapseData:
    """ Data structure to hold the data for an edge collapse operation, comparable by cost (i.e., for priority queue)"""

    def __init__(self, he: HalfEdge):
        """ Compute the edge collapse data for the given half-edge.
        Store the cost, optimal position, and quadric matrix for the edge collapse. """

        self.he = he
        # store link to Edge collapse data in both half edges
        self.he.edge_collapse_data = self
        self.he.twin.edge_collapse_data = self

        v_head = he.head
        v_tail = he.tail()

        # Combined quadric from both vertices
        self.Q = v_head.Q + v_tail.Q

        # Convert Q to numpy array for computations
        Q_np = mat4_to_numpy(self.Q)
        A = Q_np[:3, :3]
        b = Q_np[:3, 3]
        #print(A,"\n",b,"\n",Q_np)

        # Attempt to solve A * v = -b
        v_candidates = []
        solved = False
        if np.linalg.cond(A) < 1e12:
            try:
                v_solution = -np.linalg.solve(A, b)
                v_candidates.append(v_solution)
                solved = True
            except np.linalg.LinAlgError:
                solved = False
        # deficiency handles
        if not solved:
            # Use pseudoinverse , which will always give an solution
            v_solution = -np.linalg.pinv(A) @ b
            v_candidates.append(v_solution)

        # add endpoints and midpoint as fallback candidates
        head_pos = np.array((v_head.pos.x, v_head.pos.y, v_head.pos.z), dtype=float)
        tail_pos = np.array((v_tail.pos.x, v_tail.pos.y, v_tail.pos.z), dtype=float)
        midpoint = 0.5 * (head_pos + tail_pos)
        v_candidates.extend([head_pos, tail_pos, midpoint])

        # function to calculate the cost by multiplying the vTQv
        def evaluate_cost(Q_matrix: np.ndarray, position: np.ndarray) -> float:
            #to homogenous
            vec = np.append(position, 1.0)
            return float(vec.T @ Q_matrix @ vec)

        best_pos = None
        best_cost = np.inf
        for candidate in v_candidates:
            cost = evaluate_cost(Q_np, candidate)
            if cost < best_cost:
                best_cost = cost
                best_pos = candidate

        self.v_opt = glm.vec3(float(best_pos[0]), float(best_pos[1]), float(best_pos[2]))
        self.cost = best_cost

    def __lt__(self, other):
        if self.cost == other.cost:
            return id(self) < id(other)  # ensure a consistent ordering
        return self.cost < other.cost

    def __eq__(self, other):
        return id(self) == id(other)  # equal cost is not enough, must be the same edge
    

class CollapseRecord:
    """ data structure to hold the data for an edge collapse operation, for LOD tracking.
        Use a list of Faces, rather than indices, as face indices will change as we collapse."""

    def __init__(self, affected_faces: list[Face], old_indices: np.ndarray, new_indices: np.ndarray):
        self.affected_faces = affected_faces  # faces that were removed during this collapse
        self.old_indices = old_indices.copy()  # to be safe, make our own copy
        self.new_indices = new_indices.copy()

    def redo(self, faces: np.ndarray):
        """ Apply this collapse record to the given faces array."""
        for i in self.affected_faces:
            print("this is affected face index", i.index,i.he.head,i.he.tail())
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            print("swaping face index at: ", f.index)
            faces[f.index, :] = self.new_indices[i]
            
           

    def undo(self, faces: np.ndarray):
        """ Undo this collapse record on the given faces array. """
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            faces[f.index, :] = self.old_indices[i]


def build_heds(F: np.ndarray, vert_objs: list[Vertex]) -> (list[HalfEdge], list[Face]):
    """ Build a half-edge data structure from the given vertices and faces.
    Args:
        F: (num_faces, 3) array of vertex indices for each triangular face
        vert_objs: a list of vertices to set as head of the half edges
    Returns:
        List of *all* HalfEdge objects
        List of *all* Face objects
    """

    he_list = []
    face_objs = []

    # TODO: Objective 1: Build the half-edge data structure
    for fid, face in enumerate(F):
        face_obj = Face(fid)
        face_objs.append(face_obj)
        edges = []
        for head in face:
            he = HalfEdge(vert_objs[head],face_obj,None)
            edges.append(he)
            he_list.append(he)
        edges[0].next = edges[1]
        edges[1].next = edges[2]
        edges[2].next = edges[0]
    
    # Second pass: Find and link twin half edges using a dictionary for efficiency
    edge_dict = {}
    for he in he_list:
        tail_idx = he.tail().index
        head_idx = he.head.index
        # Use (min, max) as key to handle both directions
        key = (min(tail_idx, head_idx), max(tail_idx, head_idx))
        if key not in edge_dict:
            edge_dict[key] = []
        edge_dict[key].append(he)
    
    # Link twins: for each pair of half edges sharing the same vertices (in opposite directions)
    for key, edges_with_key in edge_dict.items():
        if len(edges_with_key) == 2:
            he1, he2 = edges_with_key
            # Check which direction each goes and link them as twins
            if he1.tail().index == he2.head.index and he1.head.index == he2.tail().index:
                he1.twin = he2
                he2.twin = he1 





    return he_list, face_objs