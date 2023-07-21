#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.model.node import Node
from src.material.elasto_plastic_von_mises.solid import ElastoPlasticVonMisesSolid
from src.boundary import Boundary
from method.nonlinear_fem import NonlinearFEM
from src.model.element.C3D8 import C3D8

# メインの処理
def main():
    # 節点リストを定義する
    node1 = Node(1, 0.0, 0.0, 0.0)
    node2 = Node(2, 1.0, 0.0, 0.0)
    node3 = Node(3, 2.0, 0.0, 0.0)
    node4 = Node(4, 3.0, 0.0, 0.0)
    node5 = Node(5, 0.0, 0.0, 1.0)
    node6 = Node(6, 1.0, 0.0, 1.0)
    node7 = Node(7, 2.0, 0.0, 1.0)
    node8 = Node(8, 3.0, 0.0, 1.0)
    node9 = Node(9, 0.0, 1.0, 0.0)
    node10 = Node(10, 1.0, 1.0, 0.0)
    node11 = Node(11, 2.0, 1.0, 0.0)
    node12 = Node(12, 3.0, 1.0, 0.0)
    node13 = Node(13, 0.0, 1.0, 1.0)
    node14 = Node(14, 1.0, 1.0, 1.0)
    node15 = Node(15, 2.0, 1.0, 1.0)
    node16 = Node(16, 3.0, 1.0, 1.0)
    nodes = [node1, node2, node3, node4, node5, node6, node7, node8,
             node9, node10, node11, node12, node13, node14, node15, node16]
    
    # 要素を構成する節点のコネクティビティを設定する
    nodes1 = [node1, node2, node10, node9, node5, node6, node14, node13]
    nodes2 = [node2, node3, node11, node10, node6, node7, node15, node14]
    nodes3 = [node3, node4, node12, node11, node7, node8, node16, node15]

    # 材料特性を定義する
    young = 210000.0
    poisson = 0.3
    density = 7850
    mat = ElastoPlasticVonMisesSolid(young, poisson, density)
    # 塑性硬化の条件を設定
    mat.add_stress_plastic_strain_line(400000, 0.0)
    mat.add_stress_plastic_strain_line(500000, 0.5)
    mat.add_stress_plastic_strain_line(600000, 0.7)
    mat.add_stress_plastic_strain_line(700000, 1.0)

    # 要素リストを定義する
    elem1 = C3D8(1, nodes1, mat)
    elem2 = C3D8(2, nodes2, mat)
    elem3 = C3D8(3, nodes3, mat)
    elems = [elem1, elem2, elem3]

    # 境界条件を定義する
    bound = Boundary(len(nodes))
    bound.add_SPC(1, 0.0, 0.0, 0.0)
    bound.add_SPC(5, 0.0, 0.0, 0.0)
    bound.add_SPC(9, 0.0, 0.0, 0.0)
    bound.add_SPC(13, 0.0, 0.0, 0.0)
    bound.add_force(4, 0.0, 0.0, -10000.0)
    bound.add_force(8, 0.0, 0.0, -10000.0)
    bound.add_force(12, 0.0, 0.0, -10000.0)
    bound.add_force(16, 0.0, 0.0, -10000.0)

    # 解析を実行する
    fem = NonlinearFEM(nodes, elems, bound, 10)
    fem.run()

    # 結果を出力する
    fem.output_txt("../../output/C3D8_test")

if __name__ == '__main__':
    main()
