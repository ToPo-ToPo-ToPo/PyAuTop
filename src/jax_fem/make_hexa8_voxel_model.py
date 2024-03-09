
import numpy as np
from node import Node
from C3D8 import C3D8
#=============================================================================
# 解析モデルを作成する
#=============================================================================
def run(xlength, ylength, zlength, xdiv, ydiv, zdiv):
    
    # 要素サイズ
    delta_x = xlength / float(xdiv)
    delta_y = ylength / float(ydiv)
    delta_z = zlength / float(zdiv)
                    
    # 節点座標の作成
    nodes = make_nodes(xdiv=xdiv, ydiv=ydiv, zdiv=zdiv, delta_x=delta_x, delta_y=delta_y, delta_z=delta_z)
        
    # コネクティビティの作成
    elements = make_elements(nodes=nodes, xdiv=xdiv, ydiv=ydiv, zdiv=zdiv)
                
    return nodes, elements

#=============================================================================
# 節点データを作成
# 自作型の生成を行うプログラムではjaxのjitは使用できない -> numbaのjitを使用
#=============================================================================
def make_nodes(xdiv, ydiv, zdiv, delta_x, delta_y, delta_z):
    
    # 節点座標の作成
    num_nodes = (xdiv + 1) * (ydiv + 1) * (zdiv + 1) 
    nodes = np.empty(num_nodes, dtype=object)
    for i in range(zdiv+1):
        for j in range(ydiv+1):
            for k in range(xdiv+1):
                
                # データを作成
                id = k + j*(1+xdiv) + i*(1+xdiv)*(1+ydiv)
                x = delta_x * k
                y = delta_y * j
                z = delta_z * i
                                
                # 節点の生成
                nodes[id] = Node(id+1, [x, y, z])	
                
    return nodes

#=============================================================================
# コネクティビティの作成
#=============================================================================
def make_elements(nodes, xdiv, ydiv, zdiv):
    #
    num_elements = xdiv * ydiv * zdiv
    
    #
    elements = np.empty(num_elements, dtype=object)
    counter = 0
    for i in range(zdiv):
        for j in range(ydiv):
            for k in range(xdiv):
                
                # コネクティビティの指定
                n1 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k]
                n2 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1]
                n3 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1 + (xdiv+1)]
                n4 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k     + (xdiv+1)]
                                                        
                n5 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k                + (xdiv+1)*(ydiv+1)]
                n6 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1            + (xdiv+1)*(ydiv+1)]
                n7 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1 + (xdiv+1) + (xdiv+1)*(ydiv+1)]
                n8 = nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k     + (xdiv+1) + (xdiv+1)*(ydiv+1)]
                
                # データの生成
                elements[counter] = C3D8(counter+1, np.array([n1, n2, n3, n4, n5, n6, n7, n8]))
                
                # カウンターの更新
                counter += 1
    
    return elements
    