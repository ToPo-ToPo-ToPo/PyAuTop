
import numpy as np
from node import Node
from CPS4 import CPS4
#=============================================================================
# 解析モデルを作成する
#=============================================================================
def run(xlength, ylength, xdiv, ydiv):
    
    # 要素サイズ
    delta_x = xlength / float(xdiv)
    delta_y = ylength / float(ydiv)
                    
    # 節点座標の作成
    nodes = make_nodes(xdiv=xdiv, ydiv=ydiv, delta_x=delta_x, delta_y=delta_y)
        
    # コネクティビティの作成
    elements = make_elements(nodes=nodes, xdiv=xdiv, ydiv=ydiv)
                
    return nodes, elements

#=============================================================================
# 節点データを作成
# 自作型の生成を行うプログラムではjaxのjitは使用できない -> numbaのjitを使用
#=============================================================================
def make_nodes(xdiv, ydiv, delta_x, delta_y):

    # 節点座標の作成
    num_nodes = (xdiv + 1) * (ydiv + 1)
    nodes = np.empty(num_nodes, dtype=object)
    for i in range(ydiv+1):
        for j in range(xdiv+1):
            id = j + i*(1+xdiv) 
            x = delta_x * j
            y = delta_y * i               
            # 節点の生成
            nodes[id] = Node(id+1, [x, y, 0.0])	
                
    return nodes

#=============================================================================
# コネクティビティの作成
#=============================================================================
def make_elements(nodes, xdiv, ydiv):
    #
    num_elements = xdiv * ydiv
    #
    elements = np.empty(num_elements, dtype=object)
    counter = 0
    for i in range(ydiv):
        for j in range(xdiv):       
            # コネクティビティの指定（4節点反時計回り）
            n1 = nodes[(xdiv+1) * i + j]
            n2 = nodes[(xdiv+1) * i + j + 1]
            n3 = nodes[(xdiv+1) * i + j + 1 + (xdiv+1)]
            n4 = nodes[(xdiv+1) * i + j     + (xdiv+1)]
            # データの生成
            elements[counter] = CPS4(counter+1, np.array([n1, n2, n3, n4]))
            # カウンターの更新
            counter += 1
    return elements
    