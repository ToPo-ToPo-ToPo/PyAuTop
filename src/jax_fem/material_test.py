
import numpy as np
from linear_elastic_plane_stress import LinearElasticPlaneStress

young = 1.0
poisson = 0.3
density = 1.0e-05
material = LinearElasticPlaneStress(young, poisson, density)

C = material.make_C()
print(C)

materials = np.empty(2, dtype=object)
materials[0] = LinearElasticPlaneStress(young, poisson, density)
materials[1] = LinearElasticPlaneStress(2.0 * young, poisson, density)

for material in materials:
    C = material.make_C()
    print(C)
    
