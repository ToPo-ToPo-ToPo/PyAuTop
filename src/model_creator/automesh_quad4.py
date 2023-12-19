from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.physics.node import Node
from src.physics.node_2d import Node2d
from numba import jit, f8
#=============================================================================
# ボクセルメッシュを作成する
#=============================================================================
@jit(nopython=True, cache=True)
def create_voxel_mesh_for_quad4(xdiv, ydiv, xlength, ylength):
    
    # リストの初期化
    nodes = []
    connects = []

    # 差分を計算
    delta_x = float(xlength) / float(xdiv)
    delta_y = float(ylength) / float(ydiv)

    # 節点座標の作成
    for i in range(ydiv+1):
        for j in range(xdiv+1):
            id = j + i*(1+xdiv) 
            x = delta_x * j
            y = delta_y * i
                            
            # 節点の生成
            nodes.append(Node2d(int(id+1), float(x), float(y)))	
        
    # コネクティビティの作成
    for i in range(ydiv):
        for j in range(xdiv):
            # コネクティビティの指定（4節点反時計回り）
            nodeID1 = nodes[(xdiv+1) * i + j]
            nodeID2 = nodes[(xdiv+1) * i + j + 1]
            nodeID3 = nodes[(xdiv+1) * i + j + 1 + (xdiv+1)]
            nodeID4 = nodes[(xdiv+1) * i + j     + (xdiv+1)]

            # 要素の生成
            connects.append([nodeID1, nodeID2, nodeID3, nodeID4])
                
    # 作成したデータを戻す
    return nodes, connects