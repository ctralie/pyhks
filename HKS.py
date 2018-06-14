import numpy as np 
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt 


def makeLaplacianMatrixCotWeights(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy
    
    Parameters
    ----------
    VPos : ndarray (N, 3) 
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices 
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float
    

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    #Allocate space for the sparse array storage, with 2 entries for every
    #edge for eves ry triangle (6 entries per triangle); one entry for directed 
    #edge ij and ji.  Note that this means that edges with two incident triangles
    #will have two entries per directed edge, but sparse array will sum them 
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.zeros(M*6)
    
    #Keep track of areas of incident triangles and the number of incident triangles
    IA = np.zeros(M*3)
    VA = np.zeros(M*3) #Incident areas
    VC = 1.0*np.ones(M*3) #Number of incident triangles
    
    #Step 1: Compute cotangent weights
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        dV1 = VPos[ITris[:, i], :] - VPos[ITris[:, k], :]
        dV2 = VPos[ITris[:, j], :] - VPos[ITris[:, k], :]
        Normal = np.cross(dV1, dV2)
        #Cotangent is dot product / mag cross product
        NMag = np.sqrt(np.sum(Normal**2, 1))
        cotAlpha = np.sum(dV1*dV2, 1)/NMag
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        V[shift*M*2:shift*M*2+M] = cotAlpha
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
        V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
        if shift == 0:
            #Compute contribution of this triangle to each of the vertices
            for k in range(3):
                IA[k*M:(k+1)*M] = ITris[:, k]
                VA[k*M:(k+1)*M] = 0.5*NMag
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    #Scale each row by the incident areas
    Areas = sparse.coo_matrix((VA, (IA, IA)), shape=(N, N)).tocsr()
    Areas = Areas.todia().data.flatten()
    Counts = sparse.coo_matrix((VC, (IA, IA)), shape=(N, N)).tocsr()
    Counts = Counts.todia().data.flatten()
    RowScale = sparse.dia_matrix((3*Counts/Areas, 0), L.shape)
    L = L.T.dot(RowScale).T
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def makeLaplacianMatrixUmbrellaWeights(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with "umbrella weights" (unweighted)
    by doing many operations in parallel using NumPy
    
    Parameters
    ----------
    VPos : ndarray (N, 3) 
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices 
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float
    

    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with umbrella weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    
    #Step 1: Set up umbrella entries
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    
    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    L[L > 0] = 1
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    
    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L


def getEdges(VPos, ITris):
    """
    Given a list of triangles, return an array representing the edges
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return L.nonzero()


def getLaplacianSpectrum(VPos, ITris, K):
    """
    Given a mesh, to compute first K eigenvectors of its Laplacian
    and the corresponding eigenvalues
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvectors to compute
    Returns
    -------
    (eigvalues, eigvectors): a tuple of the eigenvalues and eigenvectors
    """
    L = makeLaplacianMatrixCotWeights(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    return (eigvalues, eigvectors)


def getHeat(eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    """
    Simulate heat flow by projecting initial conditions
    onto the eigenvectors of the Laplacian matrix, and then sum up the heat
    flow of each eigenvector after it's decayed after an amount of time t
    Parameters
    ----------
    eigvalues : ndarray (K)
        Eigenvalues of the laplacian
    eigvectors : ndarray (N, K)
        An NxK matrix of corresponding laplacian eigenvectors
        Number of eigenvectors to compute
    t : float
        The time to simulate heat flow
    initialVertices : ndarray (L)
        indices of the verticies that have an initial amount of heat
    heatValue : float
        The value to put at each of the initial vertices at the beginning of time
    
    Returns
    -------
    heat : ndarray (N) holding heat values at each vertex on the mesh
    """
    N = eigvectors.shape[0]
    I = np.zeros(N)
    I[initialVertices] = heatValue
    coeffs = I[None, :].dot(eigvectors)
    coeffs = coeffs.flatten()
    coeffs = coeffs*np.exp(-eigvalues*t)
    heat = eigvectors.dot(coeffs[:, None])
    return heat

def getHKS(VPos, ITris, K, t):
    """
    Given a triangle mesh, approximate its curvature at some measurement scale
    by recording the amount of heat that remains at each vertex after a unit impulse
    of heat is applied.  This is called the "Heat Kernel Signature" (HKS)

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvalues/eigenvectors to use
    t : float
        The time scale at which to compute the HKS
    
    Returns
    -------
    hks : ndarray (N)
        A length N array of the HKS values
    """
    L = makeLaplacianMatrixUmbrellaWeights(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    eigvectors = (eigvectors**2)*np.exp(-eigvalues*t)[None, :]
    return np.sum(eigvectors, 1)

def randomlySamplePoints(VPos, ITris, NPoints, colPoints = True):
    """
    Randomly sample points by area on a triangle mesh.  This function is
    extremely fast by using broadcasting/numpy operations in lieu of loops

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    NPoints : int
        Number of points to sample
    colPoints : boolean (default True)
        Whether the points are along the columns or the rows
    
    Returns
    -------
    (Ps : NDArray (NPoints, 3) array of sampled points, 
     Ns : Ndarray (NPoints, 3) of normals at those points   )
    """
    ###Step 1: Compute cross product of all face triangles and use to compute
    #areas and normals (very similar to code used to compute vertex normals)

    #Vectors spanning two triangle edges
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    FNormals = np.cross(V1, V2)
    FAreas = np.sqrt(np.sum(FNormals**2, 1)).flatten()

    #Get rid of zero area faces and update points
    ITris = ITris[FAreas > 0, :]
    FNormals = FNormals[FAreas > 0, :]
    FAreas = FAreas[FAreas > 0]
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]

    #Compute normals
    NTris = ITris.shape[0]
    FNormals = FNormals/FAreas[:, None]
    FAreas = 0.5*FAreas
    FNormals = FNormals
    VNormals = np.zeros_like(VPos)
    VAreas = np.zeros(VPos.shape[0])
    for k in range(3):
        VNormals[ITris[:, k], :] += FAreas[:, None]*FNormals
        VAreas[ITris[:, k]] += FAreas
    #Normalize normals
    VAreas[VAreas == 0] = 1
    VNormals = VNormals / VAreas[:, None]

    ###Step 2: Randomly sample points based on areas
    FAreas = FAreas/np.sum(FAreas)
    AreasC = np.cumsum(FAreas)
    samples = np.sort(np.random.rand(NPoints))
    #Figure out how many samples there are for each face
    FSamples = np.zeros(NTris)
    fidx = 0
    for s in samples:
        while s > AreasC[fidx]:
            fidx += 1
        FSamples[fidx] += 1
    #Now initialize an array that stores the triangle sample indices
    tidx = np.zeros(NPoints, dtype=np.int64)
    idx = 0
    for i in range(len(FSamples)):
        tidx[idx:idx+FSamples[i]] = i
        idx += FSamples[i]
    N = np.zeros((NPoints, 3)) #Allocate space for normals
    idx = 0

    #Vector used to determine if points need to be flipped across parallelogram
    V3 = P2 - P1
    V3 = V3/np.sqrt(np.sum(V3**2, 1))[:, None] #Normalize

    #Randomly sample points on each face
    #Generate random points uniformly in parallelogram
    u = np.random.rand(NPoints, 1)
    v = np.random.rand(NPoints, 1)
    Ps = u*V1[tidx, :] + P0[tidx, :]
    Ps += v*V2[tidx, :]
    #Flip over points which are on the other side of the triangle
    dP = Ps - P1[tidx, :]
    proj = np.sum(dP*V3[tidx, :], 1)
    dPPar = V3[tidx, :]*proj[:, None] #Parallel project onto edge
    dPPerp = dP - dPPar
    Qs = Ps - dPPerp
    dP0QSqr = np.sum((Qs - P0[tidx, :])**2, 1)
    dP0PSqr = np.sum((Ps - P0[tidx, :])**2, 1)
    idxreg = np.arange(NPoints, dtype=np.int64)
    idxflip = idxreg[dP0QSqr < dP0PSqr]
    u[idxflip, :] = 1 - u[idxflip, :]
    v[idxflip, :] = 1 - v[idxflip, :]
    Ps[idxflip, :] = P0[tidx[idxflip], :] + u[idxflip, :]*V1[tidx[idxflip], :] + v[idxflip, :]*V2[tidx[idxflip], :]

    #Step 3: Compute normals of sampled points by barycentric interpolation
    Ns = u*VNormals[ITris[tidx, 1], :]
    Ns += v*VNormals[ITris[tidx, 2], :]
    Ns += (1-u-v)*VNormals[ITris[tidx, 0], :]

    if colPoints:
        return (Ps.T, Ns.T)
    return (Ps, Ns)

#Return VPos, VColors, and ITris without creating any structure
#(Assumes triangle mesh)
def loadOffFile(filename):
    fin = open(filename, 'r')
    nVertices = 0
    nFaces = 0
    lineCount = 0
    face = 0
    vertex = 0
    divideColor = False
    VPos = np.zeros((0, 3))
    VColors = np.zeros((0, 3))
    ITris = np.zeros((0, 3))
    for line in fin:
        lineCount = lineCount+1
        fields = line.split() #Splits whitespace by default
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        #Check section
        if nVertices == 0:
            if fields[0] == "OFF" or fields[0] == "COFF":
                if len(fields) > 2:
                    fields[1:4] = [int(field) for field in fields]
                    [nVertices, nFaces, nEdges] = fields[1:4]
                    #Pre-allocate vertex arrays
                    VPos = np.zeros((nVertices, 3))
                    VColors = np.zeros((nVertices, 3))
                    ITris = np.zeros((nFaces, 3))
                if fields[0] == "COFF":
                    divideColor = True
            else:
                fields[0:3] = [int(field) for field in fields]
                [nVertices, nFaces, nEdges] = fields[0:3]
                VPos = np.zeros((nVertices, 3))
                VColors = np.zeros((nVertices, 3))
                ITris = np.zeros((nFaces, 3))
        elif vertex < nVertices:
            fields = [float(i) for i in fields]
            P = [fields[0],fields[1], fields[2]]
            color = np.array([0.5, 0.5, 0.5]) #Gray by default
            if len(fields) >= 6:
                #There is color information
                if divideColor:
                    color = [float(c)/255.0 for c in fields[3:6]]
                else:
                    color = [float(c) for c in fields[3:6]]
            VPos[vertex, :] = P
            VColors[vertex, :] = color
            vertex = vertex+1
        elif face < nFaces:
            #Assume the vertices are specified in CCW order
            fields = [int(i) for i in fields]
            ITris[face, :] = fields[1:fields[0]+1]
            face = face+1
    fin.close()
    VPos = np.array(VPos, np.float64)
    VColors = np.array(VColors, np.float64)
    ITris = np.array(ITris, np.int32)
    return (VPos, VColors, ITris)

def saveOffFile(filename, VPos, VColors, ITris):
    """
    Save a .off file
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors.size > 0:
            fout.write(" %g %g %g"%tuple(VColors[i, :]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()

def saveHKSColors(filename, VPos, hks, ITris, cmap = 'gray'):
    """
    Save the mesh as a .coff file using a divergent colormap, where
    negative curvature is one one side and positive curvature is on the other
    """
    c = plt.get_cmap(cmap)
    x = (hks - np.min(hks))
    x /= np.max(x)
    np.array(np.round(x*255.0), dtype=np.int32)
    C = c(x)
    C = C[:, 0:3]
    saveOffFile(filename, VPos, C, ITris)

if __name__ == '__main__':
    (VPos, VColors, ITris) = loadOffFile('homer.off')
    hks = getHKS(VPos, ITris, 200, 20)
    saveHKSColors("hks.off", VPos, hks, ITris)
