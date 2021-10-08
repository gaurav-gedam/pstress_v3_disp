# Program for finite element simulation of plane stress.

# Author: Gaurav Gedam, 
# Element Type: 4 node quadrilateral isoparametric elements
# Physics: Thin plate strucure subected to in-plane loads
#   Allowed Neumann BC: constant pressure on edges and point force
#   Allowed Dirichlet BC: zero displacement on edges
#   Input: Meshed model saved as 'Mesh.msh', gmsh V2 format
#        material and cross section data in 'mat.dat'
#        Point load data in 'ptload.dat'
#        BC data in bc.dat
# Mesh in .msh v2 format

import numpy as np
from elemmats import cpressvec, stiffmat
import os
from twoDmesh import twoDmesh, physical_nodes, physical_elem_nodes

##############################################################################
#                                Pre-Processing
##############################################################################

# Read global nodal data
x, y, n1, n2, n3, n4 = twoDmesh()
numnp = x.size                  # total number of nodes
numel = n1.size               # total number of elements                  
n = 2 * numnp                 # total number of dof's

# Material properties of elements
E, nu, rho = np.loadtxt("mat.dat", unpack=True)
E = E*np.ones(numel)
nu = nu*np.ones(numel)

# To allow for single element study
if n1.size == 1: 
    n1 = np.array(n1)
    n2 = np.array(n2)
    n3 = np.array(n3)
    n4 = np.array(n4)

# Local coordinate arrays
x1 = x[n1]
y1 = y[n1]
x2 = x[n2]
y2 = y[n2]
x3 = x[n3]
y3 = y[n3]
x4 = x[n4]
y4 = y[n4]

# Element stiffness matrices
km = np.zeros((8,8, numel))
for i in range(numel):
    km[:,:,i] = stiffmat(E[i], nu[i], [x1[i], x2[i], x3[i], x4[i]], [y1[i], y2[i], y3[i], y4[i]]) #this line slows down the solution, improve the function

# Global stiffness matrix
K = np.zeros((n, n))

for h in range(numel):
    LM = np.array([2*n1[h], 2*n1[h]+1, 2*n2[h], 2*n2[h]+1, 2*n3[h], 2*n3[h]+1, 2*n4[h], 2*n4[h]+1], dtype=int)
    for i in range(8):
        for j in range(8):
            K[LM[i], LM[j]] += km [i,j,h]

# Create global load vector
f = np.zeros((2*numnp, 1))
if os.path.getsize("cpress.dat") != 0:
    
    # Read linearly varying loads and build load vector contribution due to constant pressure on element edges
    cpressentity = np.loadtxt("cpress.dat", usecols=(0), unpack=True, dtype=int)
    qx, qy = np.loadtxt("cpress.dat", usecols=(1,2), unpack=True)
    
    # Convert signle entry into np array
    if cpressentity.size == 1: 
        cpressentity = np.array([cpressentity])
        qx = np.array([qx])
        qy = np.array([qy])
    
    # Loop over all the edges with cpress, create LM array for an edge, add contri due to cpress on an elem to the global load vector 
    for i in range(cpressentity.size):   
        cpressnodes = np.array(physical_elem_nodes(cpressentity[i])).flatten()       
        for j in range(int(cpressnodes.size/2)):
            LMcpress = np.array([2*cpressnodes[2*j], 2*cpressnodes[2*j]+1, 2*cpressnodes[2*j+1], 2*cpressnodes[2*j+1]+1], dtype=int)
            f[LMcpress] += cpressvec(qx[i], qy[i], x[cpressnodes[2*j]], x[cpressnodes[2*j+1]], y[cpressnodes[2*j]], y[cpressnodes[2*j+1]])

# Read point-loads and add to the global force vector
if os.path.getsize("pointload.dat") != 0:
    # Read point loads
    ptnodes, loadtype = np.loadtxt("pointload.dat",unpack=True, dtype=int, usecols=(0,1)) # Global node number, loadtype: 0=force along 'x', 1=force along 'y'
    ptload = np.loadtxt("pointload.dat",unpack=True, usecols=(2)) # Force magnitude
    
    # Convert signle entry into np array
    if ptnodes.size == 1: ptnodes = np.array([ptnodes])
    if loadtype.size == 1: loadtype = np.array([loadtype])
    if ptload.size == 1: ptload = np.array([ptload])
    
    # Add point forces to golbal load vector
    f[2*ptnodes + loadtype] += ptload
    
# Apply Dirichlet BC
constrained_dofs = np.array([], dtype=int)
# On edges
if os.path.getsize("edgebc.dat") != 0:
    bcentity, dirn = np.loadtxt("edgebc.dat", unpack=True, dtype=int, usecols=(0,1))
    disp = np.loadtxt("edgebc.dat", unpack=True, usecols=(2))
    
    #convert single entry to numpy array
    if bcentity.size == 1: bcentity = np.array([bcentity])
    
    for i in range(bcentity.size):   
        bcnodes = np.array(physical_nodes(bcentity[i]), dtype=int).flatten()
        constrained_dofs = np.append(constrained_dofs, np.array(2*bcnodes+dirn[i], dtype=int).flatten())
        # Modify load vector to apply Dirichlet BC's
        for j in range(bcnodes.size):
            f -=K[:,int(2*bcnodes[j]+dirn[i])][:,None] * disp[i]

# On specific nodes
if os.path.getsize("nodal_bc.dat") != 0:
    cnodes, dirn2 = np.loadtxt("nodal_bc.dat", unpack=True, dtype=int, usecols=(0,1))
    disp2 = np.loadtxt("nodal_bc.dat", unpack=True, usecols=(2))
    
    #convert single entry to numpy array
    if cnodes.size == 1: 
        cnodes = np.array([cnodes])
        disp2 = np.array([disp2])
        dirn2 = np.array([dirn2])
    constrained_dofs = np.append(constrained_dofs, np.array(2*cnodes+dirn2, dtype=int).flatten())
    # Modify load vector to apply Dirichlet BC's
    for i in range(cnodes.size):
        f -= K[:,int(2*cnodes[i] + dirn2[i])][:,None] * disp2[i]

##############################################################################
# Solve for displacements
##############################################################################

# Make an array of active dof numbers

adof = np.delete(np.array(range(n)), constrained_dofs)

# Solve for displacements
from scipy.linalg import solve
u = solve(K[np.ix_(adof,adof)], f[np.ix_(adof)])

##############################################################################
#                               Post-processing
##############################################################################

# Global displacement vector
disp_vec = np.zeros((n))
disp_vec[adof] = u.flatten()
# disp_vec[constrained_dofs] = disp #edit to add imposed dsplacements

# Support Reactions (for validation)
react = np.dot(K, disp_vec)

# Node locations after deformation
xdef = x + disp_vec[np.array(range(0,2*numnp, 2))]
ydef = y + disp_vec[np.array(range(1,2*numnp, 2))]

xdef1 = xdef[n1]
ydef1 = ydef[n1]
xdef2 = xdef[n2]
ydef2 = ydef[n2]
xdef3 = xdef[n3]
ydef3 = ydef[n3]
xdef4 = xdef[n4]
ydef4 = ydef[n4]

#____________________________________________ Visualize IN PARAVIEW__________________
# Write .vtk file and open it in Paraview
# create result file and write the header
resundef = open("res_undef.vtk", "w") #undeformed geometry
header = ["# vtk DataFile Version 3.0\n", "2D-PROBLEM\n", "ASCII\n", "DATASET UNSTRUCTURED_GRID\n"]
resundef.writelines(header)

# write nodal point data
resundef.writelines(["POINTS ", str(numnp), " FLOAT\n"])

z=np.zeros((numnp))
zdef=np.zeros((numnp))

for i in range(numnp):
    resundef.writelines([str(x[i]), " ", str(y[i]), " ", str(z[i]), "\n"]) #Undeformed nodes

# write element connectivity data
resundef.writelines(["CELLS ", str(numel), " ", str(5*numel), "\n"])
for i in range(numel):
    resundef.writelines([str(4), " ", str(n1[i]), " ", str(n2[i]), " ", str(n3[i]), " ", str(n4[i]), "\n"]) #Undefomed mesh      
      
# write paraview cell type used(here 4 noded quadrilateral i.e VTK_QUAD is used)
resundef.writelines(["CELL_TYPES ", str(numel), "\n"])
resundef.write(numel * "9\n")

# Write nodal displacements
resundef.writelines(["POINT_DATA ", str(numnp), "\n", "VECTORS NODAL_DISPACEMENTS FLOAT\n"])
for i in range(numnp):
    resundef.writelines( [str(disp_vec[2*i+0]), " ", str(disp_vec[2*i+1]), " ", str(0), "\n"] )

resundef.close()