import gmsh
gmsh.initialize() 
gmsh.open('mesh.msh')
import numpy as np


def twoDmesh(): 
    # read dodal data
    nodetags = np.array(gmsh.model.mesh.getNodes(-1,-1)[0])
    coords = np.array(gmsh.model.mesh.getNodes(-1,-1)[1])
    
    coord_indices = np.array([*range(nodetags.size)], dtype=int)
    
    x = np.array(coords[3*coord_indices])
    y = np.array(coords[3*coord_indices+1])
    
    # read element data
    
    elemtags = np.array(gmsh.model.mesh.getElements(dim=2,tag=-1)[1]).flatten()
    elem_nodetags = np.array(gmsh.model.mesh.getElements(dim=2,tag=-1)[2]).flatten()
    node_indices = np.array([*range(elemtags.size)], dtype=int)
    
    n1 = np.array(elem_nodetags[4*node_indices]).flatten()
    n2 = np.array(elem_nodetags[4*node_indices+1]).flatten()
    n3 = np.array(elem_nodetags[4*node_indices+2]).flatten()
    n4 = np.array(elem_nodetags[4*node_indices+3]).flatten()
    
    return(x, y, n1-1, n2-1, n3-1, n4-1) # 1 is subtracted to start indices from zero

def physical_elem_nodes(physical_tag):
    # import gmsh
    # import numpy as np
    # gmsh.open('mesh.msh')
    physical_nodetags = np.array(gmsh.model.mesh.getElements(dim=1,tag=physical_tag)[2]).flatten()
    
    return physical_nodetags-1,  #1 is subtracted to start indices from zero
    
def physical_nodes(physical_tag):
    # import gmsh
    # import numpy as np
    # gmsh.open('mesh.msh')
    element_node_tags = np.array(gmsh.model.mesh.getElements(dim=1,tag=physical_tag)[2]).flatten()
    physical_nodetags = np.unique(element_node_tags)
   
    return physical_nodetags-1,  #1 is subtracted to start indices from zero 