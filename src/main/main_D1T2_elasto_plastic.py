
from node_1d import Node1d
from Material import Material
from Boundary1d import Boundary1d
from FEM1d import FEM1d
from d1t2 import d1t2

# メインの処理
def main():
    # 節点を定義する
    node1 = Node1d(1, 0.0)
    node2 = Node1d(2, 100.0)
    nodes = [node1, node2]
    nodes1 = [node1, node2]

    # 材料情報を定義する
    area = 1.0
    young = 210e+03
    mat = Material(young, area)
    mat.addStressPStrainLine(200e3, 0.0)
    mat.addStressPStrainLine(250e3, 0.2)
    mat.addStressPStrainLine(290e3, 0.4)
    mat.addStressPStrainLine(320e3, 0.6)
    mat.addStressPStrainLine(340e3, 0.8)
    mat.addStressPStrainLine(350e3, 1.0)

    # 要素を定義する
    elem1 = d1t2(1, nodes1, mat)
    elems = [elem1]

    # 境界条件を定義する
    bound = Boundary1d(len(nodes))
    bound.addSPC(1, 0.0)
    bound.addForce(2, 345e3)

    # 解析を行う
    fem = FEM1d(nodes, elems, bound, 10)
    fem.impAnalysis()
    fem.outputTxt("../../output/D1T2_test")  

main()
