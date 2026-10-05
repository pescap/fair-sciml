import re
import sys

path = sys.argv[1]
src = open(path).read()

src = src.replace("import numpy as np\nimport torch\n", "import numpy as np\nimport scipy.sparse as sp\nimport torch\n", 1)

src = src.replace(
    "    return np.kron(P_y, P_x)\n",
    "    return sp.kron(sp.csr_matrix(P_y), sp.csr_matrix(P_x), format=\"csr\")\n", 1)

old_loop = src[src.index("        for L, (nx, ny) in enumerate(dims):"):src.index("        # ---- 3. prolongation")]
new_loop = '''        for L, (nx, ny) in enumerate(dims):
            A = self._assemble_operator(nx, ny, ngp)
            A_coo = _scipy_to_torch_coo(A)
            self.register_buffer(f"A_indices_{L}", A_coo.indices())
            self.register_buffer(f"A_values_{L}", A_coo.values())
            self.register_buffer(f"diag_{L}", torch.from_numpy(A.diagonal().copy()).float())
            if L == n_levels - 1:
                self.register_buffer("A_inv_coarsest", torch.from_numpy(np.linalg.inv(A.toarray())).float())

'''
src = src.replace(old_loop, new_loop, 1)

old_prol = src[src.index("            P_t = torch.from_numpy(_build_2d_prolongation"):src.index("    def _assemble_operator")]
new_prol = '''            P_s = _build_2d_prolongation(nx_f, ny_f, nx_c, ny_c)
            P_coo = _scipy_to_torch_coo(P_s)
            R_coo = _scipy_to_torch_coo(P_s.T.tocsr())
            self.register_buffer(f"P_indices_{L}", P_coo.indices())
            self.register_buffer(f"P_values_{L}", P_coo.values())
            self.register_buffer(f"R_indices_{L}", R_coo.indices())
            self.register_buffer(f"R_values_{L}", R_coo.values())

'''
src = src.replace(old_prol, new_prol, 1)

old_asm = src[src.index("        mesh = structured_quad_mesh(nx=nx, ny=ny)\n        A = LaplaceElementAssembler"):src.index("    # -------------------- sparse tensor reconstruction")]
new_asm = '''        mesh = structured_quad_mesh(nx=nx, ny=ny)
        A = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=ngp)(mesh.points)
        Op = self.a2 * A.to_scipy_coo().tocsr().astype(np.float64)
        if self.c != 0.0:
            M = MassElementAssembler.from_mesh(mesh, quadrature_order=ngp)(mesh.points)
            Op = Op + self.c * M.to_scipy_coo().tocsr().astype(np.float64)
        mask = mesh.boundary_mask.cpu().numpy().astype(bool)
        keep = sp.diags((~mask).astype(np.float64))
        Op = keep @ Op @ keep + sp.diags(mask.astype(np.float64))
        Op.eliminate_zeros()
        return Op.tocsr()

'''
src = src.replace(old_asm, new_asm, 1)

src = src.replace(
    "class GeometricMultigrid(Preconditioner):",
    '''def _scipy_to_torch_coo(S):
    S = S.tocoo()
    idx = torch.from_numpy(np.vstack([S.row, S.col]).astype(np.int64))
    val = torch.from_numpy(S.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, S.shape).coalesce()


class GeometricMultigrid(Preconditioner):''', 1)

open(path, "w").write(src)
