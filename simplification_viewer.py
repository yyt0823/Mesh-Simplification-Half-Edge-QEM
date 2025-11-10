from doctest import debug
import trimesh
import numpy as np
from pathlib import Path
import moderngl as mgl
from sortedcontainers import SortedList
from pyglm import glm
from PyQt5 import QtOpenGL
from heds import HalfEdge, Vertex, EdgeCollapseData, build_heds, CollapseRecord
from moderngl_text.text_renderer import TextRenderer


class SimplificationViewer(QtOpenGL.QGLWidget):
    """ OpenGL widget with simple viewing controls """

    def __init__(self):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        super(SimplificationViewer, self).__init__(fmt, None)
        self.update_UI_callback = None  # notify UI of changes (e.g., LOD slider)
        self.keyboard_callback = None  # make keyboard usable
        self.draw_quadrics = False
        self.draw_face_IDs = False
        self.draw_vertex_IDs = False
        self.draw_current_he = True
        self.scale_with_LOD = False        
        self.mesh_wireframe = False
        self.current_LOD = 0
        self.max_LOD = 0  # the number of simplification levels available
        self.collapse_history = []  # keep track of collapse records for undoing
        self.last_mouse_pos = None
        self.R = glm.mat4(1)  # current rotation matrix for view
        self.d = 4.0  # current distance of the camera

        self.vao, self.vbo, self.ibo = None, None, None  # Will be created when mesh is loaded
        self.text_renderer = None # used for rendering text labels on verts and faces
        self.ctx = None
        self.prog_tris = None  # Mesh shader
        self.prog_lines = None  # Line shader
        self.width, self.height = None, None
        self.P, self.V, self.S = None, None, None  # Projection, View, Scale matrices

        # For visualizing half edges
        self.he_geom, self.he_vbo, self.he_vao, self.current_he = None, None, None, None
        # General data
        self.indices, self.faces, self.verts = None, None, None
        self.vert_objs, self.face_objs = None, None
        self.sorted_edge_list = SortedList()

    def set_update_UI_callback(self, callback):
        self.update_UI_callback = callback

    def set_keyboard_callback(self, callback):
        self.keyboard_callback = callback

    def keyPressEvent(self, event):
        if self.keyboard_callback is not None:
            self.keyboard_callback(event)

    def initializeGL(self):
        self.ctx = mgl.create_context()
        self.text_renderer = TextRenderer(self.ctx)
        current_dir = Path(__file__).parent  # glsl folder in same directory as this code
        self.setup_mesh_shaders_and_buffers(current_dir)
        self.setup_half_edge_shaders_and_buffers(current_dir)
        mesh_file = current_dir / 'data' / 'icosphere.obj'
        self.load_mesh_from_file(str(mesh_file))

    def resizeGL(self, width, height):
        self.ctx.viewport = (0, 0, width, height)
        self.width = width
        self.height = height

    def paintGL(self):
        self.update_matrices()
        self.ctx.clear(0, 0, 0, 1)  # clear the whole drawing surface
        self.ctx.enable(mgl.DEPTH_TEST)  # always use depth test!
        self.ctx.enable(mgl.BLEND)
        # draw the mesh at the current LOD
        # no culling for triangles (helps to see backside if using boundaries (or for bugs))
        self.ctx.disable(mgl.CULL_FACE)
        if self.mesh_wireframe:
            self.ctx.wireframe = self.mesh_wireframe
            self.prog_tris['u_use_lighting'].value = False
        self.prog_tris['u_color'].value = (0.8, 0.7, 0.6, 1 - 0.5 * self.mesh_wireframe)  # set the color uniform
        if len(self.face_objs) == 0:
            self.vao.render()  # HEDS not built yet, just draw everything!
        else:
            self.vao.render( vertices = 3 * (len(self.face_objs) - 2*self.current_LOD) )# skip two faces for each LOD    

        self.ctx.wireframe = False
        self.prog_tris['u_use_lighting'].value = True
        # as selected in the UI, draw half edge, vertex IDs, face IDs
        if self.draw_current_he and self.current_LOD == self.max_LOD:
            self.ctx.enable(mgl.CULL_FACE)  # ray traced cylinders need backface culling
            self.ctx.cull_face = 'back'
            self.he_vao.render()  # draw the half-edge
            self.ctx.disable(mgl.CULL_FACE)  # ray traced cylinders need backface culling
        if self.draw_vertex_IDs:
            # New verts always added to end, so go to end of list if at highest LOD, otherwise stop before new verts
            k = self.max_LOD - self.current_LOD
            for v in self.vert_objs if k == 0 else self.vert_objs[:-k]:
                if v.removed_at_level is not None and v.removed_at_level < self.current_LOD:
                    continue
                v.draw_debug(self.P, self.V, self.text_renderer)
        if self.draw_face_IDs:
            for i in range( len(self.face_objs) - 2*self.current_LOD ):
                self.face_objs[i].draw_debug(self.P, self.V, self.faces, self.vert_objs, self.text_renderer)

    def get_vertex_count(self):
        count = 0
        k = self.max_LOD - self.current_LOD
        for v in self.vert_objs if k == 0 else self.vert_objs[:-k]:
            if v.removed_at_level is not None and v.removed_at_level < self.current_LOD:
                continue
            count += 1
        return count

    def get_face_count(self):
        return len(self.face_objs) - 2 * self.current_LOD

    def setup_mesh_shaders_and_buffers(self, current_dir):
        self.prog_tris = self.ctx.program(
            vertex_shader=open(current_dir / 'glsl' / 'tris-vert.glsl').read(),
            geometry_shader=open(current_dir / 'glsl' / 'tris-geom.glsl').read(),
            fragment_shader=open(current_dir / 'glsl' / 'tris-frag.glsl').read())
        self.prog_tris['u_color'].value = (0.8, 0.7, 0.6, 1.0)  # set the color uniform
        self.prog_tris['u_use_lighting'].value = True
        self.prog_tris['u_light_pos'].value = (15, 15, 15)  # light position in view coordinates

    def setup_half_edge_shaders_and_buffers(self, current_dir):
        self.prog_lines = self.ctx.program(
            vertex_shader=open(current_dir / 'glsl' / 'fancyline-vert.glsl').read(),
            geometry_shader=open(current_dir / 'glsl' / 'fancyline-geom.glsl').read(),
            fragment_shader=open(current_dir / 'glsl' / 'fancyline-frag.glsl').read())
        self.prog_lines['u_color'].value = (1.0, 0.0, 0.0, 1.0)
        self.prog_lines['u_light_pos'].value = (15, 15, 15)  # light position in view coordinates
        self.prog_lines['u_radius'].value = 0.02  # radius of the line in world coordinates

        # set up a half edge... and perhaps a geometry shader for it to draw lines as cylinders?
        self.he_geom = np.zeros((3, 3), dtype='f4')  # 3 vertices to form a half edge
        self.he_vbo = self.ctx.buffer(self.he_geom.tobytes())
        self.he_vao = self.ctx.vertex_array(self.prog_lines, [(self.he_vbo, '3f', 'in_position')], mode=mgl.LINE_STRIP)

    def load_mesh_from_file(self, filename: str):
        mesh = trimesh.load_mesh(filename)
        self.faces = mesh.faces  # this is easy because we have only triangles!
        self.indices = mesh.faces.flatten()
        self.verts = mesh.vertices  # shape is (num_verts, 3)
        # center and scale the mesh to fit in a -1 to 1 cube
        centroid = np.mean(self.verts, axis=0)
        self.verts -= centroid
        scale = np.max(np.linalg.norm(self.verts, axis=1))
        if scale > 1e-6:
            self.verts /= scale
        # Make vertex objects, initially without half edges,
        # 	and only for the current vertices (this list will grow as we collapse edges)
        self.vert_objs = [Vertex(i, pos, None) for i, pos in enumerate(self.verts)]
        # build the half edge data structure
        half_edges, self.face_objs = build_heds(self.faces, self.vert_objs)

        # NOTE: reserve an appropriate amount of space in the vertex buffer
        # Every collapse removes one vertex, 2 faces, and 3 edges.
        # won't go past a tetrahedron, so expect half of num faces - 4 
        num_collapses_expected = (self.faces.shape[0] - 4) // 2  # each collapse removes 2 faces

        self.verts = np.vstack((self.verts, np.zeros((num_collapses_expected, 3), dtype='f4')))

        self.vbo = self.ctx.buffer(self.verts.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(self.faces.flatten().astype('i4').tobytes())
        self.vao = self.ctx.vertex_array(self.prog_tris, [(self.vbo, '3f', 'in_position')], self.ibo, mode=mgl.TRIANGLES)

        # once the heds is initialized, compute the quadrics for each vertex, and the edge collapse costs
        if half_edges:
            for v in self.vert_objs:
                v.compute_Q()
                v.compute_debug_viz_data()  # while we're at it, collect some debug viz data

            self.compute_edge_collapse_costs(half_edges)

            self.current_he = half_edges[0]
            self.update_half_edge_geometry()

        self.current_LOD = 0
        self.max_LOD = 0  # the number of simplification levels available
        self.collapse_history = []  # keep track of collapse records for undoing
        self.update_UI_callback()  # updates the LOD slider

    def update_half_edge_geometry(self):
        print("-------",self.current_he)
        v0 = self.current_he.next.next.head.pos
        v1 = self.current_he.head.pos
        v2 = self.current_he.next.head.pos
        center = self.current_he.face.get_center()
        normal = self.current_he.face.get_normal() * glm.length(center - v0) * 0.1
        self.he_geom[0] = normal + 0.9 * v0 + 0.1 * center
        self.he_geom[1] = normal + 0.9 * v1 + 0.1 * center
        self.he_geom[2] = normal + 0.9 * (0.8 * v1 + 0.2 * v2) + 0.1 * center
        self.he_vbo.write(self.he_geom.tobytes())

    def update_matrices(self):
        self.P = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1, 100.0)
        self.S = glm.mat4(1)
        if self.scale_with_LOD:
            max_lod = len(self.face_objs) // 2
            min_scale = 0.01  # 1% at maximum LOD
            # Solve: min_scale = base^max_lod  =>  base = min_scale^(1/max_lod)
            base = min_scale ** (1.0 / max_lod)
            scale_factor = base ** self.current_LOD
            self.S = glm.scale(glm.mat4(1), glm.vec3(scale_factor, scale_factor, scale_factor))
        self.V = glm.lookAt(glm.vec3(0, 0, self.d), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0)) * self.R * self.S        
        self.prog_tris['u_mv'].write(self.V)
        self.prog_tris['u_mvp'].write(self.P * self.V)
        self.prog_lines['u_view'].write(self.V)
        self.prog_lines['u_proj'].write(self.P)

    def mousePressEvent(self, event):
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        new_x, new_y = event.x(), event.y()
        rx = glm.rotate(glm.mat4(1), (new_y - self.last_mouse_pos[1]) * 0.002, glm.vec3(1, 0, 0))
        ry = glm.rotate(glm.mat4(1), (new_x - self.last_mouse_pos[0]) * 0.002, glm.vec3(0, 1, 0))
        self.R = ry * rx * self.R
        self.last_mouse_pos = (new_x, new_y)

    def wheelEvent(self, event):
        self.d *= np.power(1.1, event.angleDelta().y() / 120)

    def next_half_edge(self):
        if self.current_he is None or self.current_he.twin is None:
            return
        self.current_he = self.current_he.next
        self.update_half_edge_geometry()

    def twin_half_edge(self):
        if self.current_he is None or self.current_he.twin is None:
            return
        self.current_he = self.current_he.twin
        self.update_half_edge_geometry()

    def collapse_current_half_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return
        # discard current half edge form queue if it is there (note that this is different from remove)
        self.sorted_edge_list.discard(self.current_he.edge_collapse_data)
        vtail = self.current_he.tail()
        vhead = self.current_he.head
        vstar = (vtail.pos + vhead.pos) / 2
        self.current_he = self.collapse(self.current_he, vstar)
        self.update_half_edge_geometry()

    def jump_to_best_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return
        edge = self.sorted_edge_list[0]
        self.current_he = edge.he
        self.update_half_edge_geometry()

    def collapse_best_edge(self):
        if len(self.sorted_edge_list) <= 6:  # stop when we reach a tetrahedron
            print("No more edges to collapse")
            return
        edge = self.sorted_edge_list.pop(0)  # get and remove the best edge
        self.current_he = self.collapse(edge.he, edge.v_opt, edge.cost)
        self.update_half_edge_geometry()

    def collapse_all_in_order(self):
        # collapse all the edges in order, best edge each time,
        # 	until we get to a tetrahedron (or otherwise simplest possible mesh)
        while not len(self.sorted_edge_list) <= 6:
            edge = self.sorted_edge_list.pop(0)  # get and remove the best edge
            self.current_he = self.collapse(edge.he, edge.v_opt, edge.cost)
        self.update_half_edge_geometry()

    def set_LOD(self, level: int):
        """ set the current LOD to the given level, updating the mesh faces data accordingly """
        if level < 0 or level > self.max_LOD:
            print("Invalid LOD level:", level)
            return
        while self.current_LOD < level:
            self.collapse_history[self.current_LOD].redo(self.faces)
            self.current_LOD += 1
        while self.current_LOD > level:
            self.current_LOD -= 1
            self.collapse_history[self.current_LOD].undo(self.faces)
        self.ibo.write(self.faces.flatten().astype('i4').tobytes())

    def compute_edge_collapse_costs(self, half_edges: list[HalfEdge]):
        """ compute the cost of collapsing each edge, and the optimal position for the collapse
            Store these in a sorted list for O(log(n)) addition and removal, and O(1) query of best edge.
            This is ONLY called on the first initialization, when we have a list of all half edges! """
        self.sorted_edge_list = SortedList()
        for he in half_edges:
            if he.edge_collapse_data is not None:
                continue  # already computed
            edge = EdgeCollapseData(he)
            self.sorted_edge_list.add(edge)

    def collapse_will_be_bad(self, he: HalfEdge) -> bool:
        """ check if collapsing this half-edge will create problems (e.g., more than 2 common verts in the 1-rings) """

        # TODO: Objective 4: Check if collapsing this half-edge will create problems


        return False

    def collapse(self, he: HalfEdge, vstar: glm.vec3, cost: float = 0.0) -> HalfEdge:
        """ collapse the given half-edge.
            If the collapse will not be bad, collapse the edge by:
            - Making a new Vertex and add to the end of the buffer (and updating the vbo accordingly)
            - Moving the collapsed faces to the end of the face_obj list (and updating the faces numpy array accordingly)
            - updating the half edge data structure TODO
            - updating the computed edge collapse costs for affected half-edges TODO
            - updating the collapse records  TODO

            Args:
                he: the hald-edge to collapse
                vstar: the position of the new vertex
                cost: cost of the new vertex
            Returns:
                 One of the half-edges with the new vertex as head.
        """

        # ensure we are at the coarest LOD
        self.set_LOD(self.max_LOD)

        if self.collapse_will_be_bad(he):
            print("Bad collapse detected, skipping")
            return he

        # flag old verts as removed at this LOD level (for debug viz)
        he.head.removed_at_level = self.max_LOD
        he.tail().removed_at_level = self.max_LOD

        # Create the new vertex at END OF THE LIST.
        # Set index to be next in the list, and give a half edge that will exist after the collapse
        new_vertex = Vertex(len(self.vert_objs), vstar, he.next.twin)
        new_vertex.cost = cost
        self.vert_objs.append(new_vertex)
        new_vertex.Q = he.head.Q + he.tail().Q
        # put it into the verts array, and into the vertex buffer object
        self.verts[new_vertex.index] = np.array((vstar.x, vstar.y, vstar.z), dtype='f4')
        self.vbo.write(vstar.to_bytes(), offset=3 * new_vertex.index * 4)


        # TODO: Objective 2: Collapse the given half-edge.  See notes from class!
        # NOTE: removed faces must be swapped with those at the end of face_objs list!
        # see also that you migth want to complete objective 3 to see the results as
        # the index buffer needs to be updated to correctly draw the mesh after the 
        # collapse, and that is done in objective 3 below.


        # objective 3 tracing
        afs = []
        ofs = []
        nfs = []

            
        start = he.next.twin
        current = start
        # head
        while True:
            face = current.face
            afs.append(face)
            vcurrent = current
            vertices_list = []
            while True:
                vertices_list.append(vcurrent.head.index)
                vcurrent = vcurrent.next
                if vcurrent == current:
                    break
            ofs.append(vertices_list)
            current = current.next.twin
            if current.next.twin.next.twin == start:
                break    
        
        # tail

        start = he.twin.next.twin
        current = start
        while True:
            face = current.face
            afs.append(face)
            vcurrent = current
            vertices_list = []
            while True:
                vertices_list.append(vcurrent.head.index)
                vcurrent = vcurrent.next
                if vcurrent == current:
                    break
            ofs.append(vertices_list)
            current = current.next.twin
            if current == start:
                break    

       
            




        


        def find_inward_edge(he: HalfEdge) -> HalfEdge:
            current = he
            while current.next is not he:
                current = current.next
            return current

        # arg: half edge , new vertex vstar, cost
        #step 1 find the 4 outer edge
        oe1 = he.next.twin
        oe2 = (he.next).next.twin
        oe3 = he.twin.next.twin
        oe4 = (he.twin.next).next.twin
        # set them to be twins
        oe1.twin = oe2
        oe2.twin = oe1
        oe3.twin = oe4
        oe4.twin = oe3

        # step 2: fix vertex that store the deleted edge
        # find vl and vr and set assign their he.twin to their he
        v1 = he.next.head
        v2 = he.twin.next.head
        v1.he = find_inward_edge(he.next.twin)
        v2.he = find_inward_edge(he.twin.next.twin)

        # step 3 : create new vertex and link use the new_vertex above
        
        start = he.next.twin
        current = start
        while True:
            current.head = new_vertex    
            current = current.next.twin
            if current.next.twin is start:
                # Process one more time: the current edge before breaking
                current.head = new_vertex
                break
        
        # Swap removed faces to end of face_objs
        face1 = he.face
        face2 = he.twin.face
        n = len(self.face_objs)
        pos1 = n - 2 * self.current_LOD - 1
        pos2 = n - 2 * self.current_LOD - 2
        face3 = self.face_objs[pos1]
        face4 = self.face_objs[pos2]
        
        # Swap face1 with face3 (both in face_objs and self.faces)
        idx1 = face1.index
        self.face_objs[idx1], self.face_objs[pos1] = face3, face1
        face1.index = pos1
        face3.index = idx1
        # Swap corresponding rows in self.faces
        self.faces[idx1], self.faces[pos1] = self.faces[pos1].copy(), self.faces[idx1].copy()

        # Swap face2 with face4 (both in face_objs and self.faces)
        idx2 = face2.index
        self.face_objs[idx2], self.face_objs[pos2] = face4, face2
        face2.index = pos2
        face4.index = idx2
        # Swap corresponding rows in self.faces
        self.faces[idx2], self.faces[pos2] = self.faces[pos2].copy(), self.faces[idx2].copy()

        


        swaped = []
        new_face_he = self.face_objs[face3.index].he
        new_vertices_list = []
        current = new_face_he

        while True:
            new_vertices_list.append(current.head.index)
            current = current.next
            if current == new_face_he:
                break
        swaped.append(new_vertices_list)
            

        new_face_he = self.face_objs[face4.index].he
        new_vertices_list = []
        current = new_face_he

        while True:
            new_vertices_list.append(current.head.index)
            current = current.next
            if current == new_face_he:
                break
        swaped.append(new_vertices_list)
        
        

        
    


        # TODO: Objective 6: Maintain the sorted list of edge collapse costs
        # After collapsing an edge, the quadric error metrics of the new vertex is set,
        # so we need to do some careful work to make sure the sorted list is updated to
        # have updated edge collapse data for the half-edges around the new vertex.
        # (best to remove all, and then re-add newly computed versions)
        


        # TODO: Objective 3: Undo / Redo by making and collecting collapse records 
        # helper
        
            

        # TODO: You need to fill in the correct data for the CollapseRecord here
        affected_faces = [] # list of N Face objects affected by this collapse (typically around 8 faces)
        old_faces = np.array([]) # Nx3 array of old face indices before the collapse
        new_faces = np.array([]) # Nx3 array of new face indices after the collapse


        
        affected_faces = afs
        for i in affected_faces:
            print("affected",i)
        old_faces = np.array(ofs)

        new_edge = new_vertex.he
        
        current = new_edge
        while True:
            vcurrent = current
            vertices_list = []
            while True:
                vertices_list.append(vcurrent.head.index)
                
                vcurrent = vcurrent.next
                if vcurrent == current:
                    break
            print("bug",vertices_list)
            nfs.append(vertices_list)
            current = current.next.twin
            if current == start:
                break  
       

        nfs = nfs + swaped
        new_faces = np.array(nfs)


        for i in new_faces:
            print("debug1",i)
        


        





        # collapse record tells us how to move to the next LOD, or likewise, how to undo
        cr = CollapseRecord(affected_faces, old_faces, new_faces)
        self.collapse_history.append(cr)
        self.current_LOD += 1
        self.max_LOD += 1

        

        cr.redo(self.faces) # THIS APPLIES THE COLLAPSE TO THE FACES numpy array for drawing with opengl
        for i in self.faces:
            print("faces", i)
        for i in self.face_objs:
            print("face_objs", i)     
        self.ibo.write(self.faces.flatten().astype('i4').tobytes()) # update the index buffer object

        # with everything all hooked up, compute the debug viz data for the new vertex
        new_vertex.compute_debug_viz_data()

        # notify the UI that LOD has changed
        self.update_UI_callback()
        return new_vertex.he  # return one of the half-edges with the new vertex as head
    
    