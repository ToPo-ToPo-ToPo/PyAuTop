
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.physics.node_1d import Node1d
from src.material.elasto_plastic_von_mises.truss import ElastoPlasticVonMisesTruss
from src.boundary_1d import Boundary1d
from src.method.nonlinear_fem_1d import FEM1d
from src.physics.element.D1T2 import D1T2

#=============================================================================
# メインの処理
#=============================================================================
def main():
    # 節点を定義する
    node1 = Node1d(1, 0.0)
    node2 = Node1d(2, 100.0)
    nodes = [node1, node2]

    # 要素のコネクティビティの定義
    nodes1 = [node1, node2]

    # 材料情報を定義する
    young = 210e+03
    mat = ElastoPlasticVonMisesTruss(young)
    mat.add_stress_plastic_strain_line(200e+03, 0.0)
    mat.add_stress_plastic_strain_line(250e+03, 0.2)
    mat.add_stress_plastic_strain_line(290e+03, 0.4)
    mat.add_stress_plastic_strain_line(320e+03, 0.6)
    mat.add_stress_plastic_strain_line(340e+03, 0.8)
    mat.add_stress_plastic_strain_line(350e+03, 1.0)

    # 要素を定義する
    area = 1.0
    elem1 = D1T2(1, nodes1, mat, area)
    elems = [elem1]

    # 境界条件を定義する
    bound = Boundary1d(len(nodes))
    bound.add_SPC(1, 0.0)
    bound.add_force(2, 345e+03)

    # 解析を行う
    fem = FEM1d(nodes, elems, bound, 10)
    fem.analysis()
    fem.output_txt("../../output/D1T2_test")  

#=============================================================================
#
#=============================================================================
if __name__ == '__main__':
    main()
