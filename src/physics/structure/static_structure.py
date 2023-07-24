from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.physics.node import Node
from src.material.elastic.solid import ElasticSolid
from src.material.elasto_plastic_von_mises.solid import ElastoPlasticVonMisesSolid
from src.boundary import Boundary
from physics.element.C3D8_Bbar import C3D8Bbar
#=============================================================================
#
#=============================================================================
class StaticStructure:
    
    def __init__(self, id):
        
        # 初期化する
        self.id = id
        self.input_file_path = "/input/structural_input.dat"
        self.output_file_path = "/output/structral_result.dat"

        self.nodes = []
        self.connects = []
        self.elems = []

    #---------------------------------------------------------------------
    # メッシュ情報を作成する
    # 戻り値 nodes, connects
    #---------------------------------------------------------------------
    def create_model_mesh(self):

        # 解析条件のinputファイルを開く
        input_f = open(parent_dir + self.input_file_path, 'r')

        # ファイルの読み込みを行う
        while True:

            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = input_f.readline()
            str = str.rstrip('\n')

            # 空白かどうかをチェックし、該当すれば読み取りを終了する
            if str == '':
                break

            # 解析メッシュを作成する
            if str == '---------------------------------------------------------------------|Geometry|':

                # コメントアウトされていない文字列を探索する
                while True:
                    # 文字列を1行読み込み、末尾の改行'\n'を取り除く
                    str = input_f.readline()
                    str.rstrip('\n')
                    
                    # 空白行でループを脱出する
                    if str == '-':
                        break

                    # タブで文字列を分解する
                    str_list = str.split(' ')

                    # コメントアウトされていないか確認する
                    if str_list[0] != 'Mesh:':
                        continue
                    
                    # コメントアウトされていない場合
                    mesh_type = str_list[1]
                    xlength = float(str_list[3])
                    ylength = float(str_list[5])
                    zlength = float(str_list[7])
                    xdiv = int(str_list[9])
                    ydiv = int(str_list[11])
                    zdiv = int(str_list[13])
                    
                    delta_x = float(xlength) / float(xdiv)
                    delta_y = float(ylength) / float(ydiv)
                    delta_z = float(zlength) / float(zdiv)
                    
                    # ボクセルメッシュを作成する
                    # 節点座標の作成
                    for i in range(zdiv+1):
                        for j in range(ydiv+1):
                            for k in range(xdiv+1):
                                id = k + j*(1+xdiv) + i*(1+xdiv)*(1+ydiv)
                                x = delta_x * k
                                y = delta_y * j
                                z = delta_z * i
                                
                                # 節点の生成
                                self.nodes.append(Node(int(id+1), float(x), float(y), float(z)))	
        
                    # コネクティビティの作成
                    for i in range(zdiv):
                        for j in range(ydiv):
                            for k in range(xdiv):
                                
                                # コネクティビティの指定
                                nodeID1 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k]
                                nodeID2 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1]
                                nodeID3 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1 + (xdiv+1)]
                                nodeID4 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k     + (xdiv+1)]
                                                        
                                nodeID5 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k                + (xdiv+1)*(ydiv+1)]
                                nodeID6 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1            + (xdiv+1)*(ydiv+1)]
                                nodeID7 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k + 1 + (xdiv+1) + (xdiv+1)*(ydiv+1)]
                                nodeID8 = self.nodes[(xdiv+1)*(ydiv+1)*i + (xdiv+1) * j + k     + (xdiv+1) + (xdiv+1)*(ydiv+1)]

                                # 要素の生成
                                self.connects.append([nodeID1, nodeID2, nodeID3, nodeID4, 
                                                      nodeID5, nodeID6, nodeID7, nodeID8])

                    #self.nodes, self.connects = auto_mesh(mesh_type, xlength, ylength, zlength, xdiv, ydiv, zdiv)
        
        # inputファイルを閉じる
        input_f.close()

        return self.nodes, self.connects

    #---------------------------------------------------------------------
    # コンポーネント情報をまとめたリストを作成する
    # 戻り値 commponent_list
    #---------------------------------------------------------------------    
    def create_commponent(self):
        
        # 解析条件のinputファイルを開く
        input_f = open(parent_dir + self.input_file_path, 'r')
        
        # 初期化
        commponent_list = []

        # ファイルの読み込みを行う
        while True:

            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = input_f.readline()
            str = str.rstrip('\n')

            # 空白かどうかをチェックし、該当すれば読み取りを終了する
            if str == '':
                break


            # コンポーネント情報を取得する
            if str == '--------------------------------------------------------------------|Component|':
                
                # コメントアウトされていない文字列を探索する
                while True:
                    # 文字列を1行読み込み、末尾の改行'\n'を取り除く
                    str = input_f.readline()
                    str = str.rstrip('\n')
                    
                    # 空白行でループを脱出する
                    if str == '-':
                        break

                    # タブで文字列を分解する
                    str_list = str.split(' ')

                    # コメントアウトされていないか確認する
                    if str_list[0] != 'Component:':
                        continue

                    # コメントアウトされていない場合
                    commponent_list.append([str_list[1], str_list[3], str_list[5], str_list[7]])
        
        # inputファイルを閉じる
        input_f.close()

        return commponent_list

    #---------------------------------------------------------------------
    # 材料モデルのlistを作成する
    # 戻り値 material_list
    #---------------------------------------------------------------------
    def create_material_model(self):
        
        # 解析条件のinputファイルを開く
        input_f = open(parent_dir + self.input_file_path, 'r')
        
        # 初期化
        material_list = []

        # ファイルの読み込みを行う
        while True:

            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = input_f.readline()
            str = str.rstrip('\n')

            # 空白かどうかをチェックし、該当すれば読み取りを終了する
            if str == '':
                break

            # 材料物性を取得し、要素を作成する
            if str == '---------------------------------------------------------------|Material_Model|':
                
                # コメントアウトされていない文字列を探索する
                while True:
                    # 文字列を1行読み込み、末尾の改行'\n'を取り除く
                    str = input_f.readline()
                    str = str.rstrip('\n')
                    
                    # 空白行でループを脱出する
                    if str == '-':
                        break

                    # タブで文字列を分解する
                    str_list = str.split(' ')

                    # コメントアウトされていないか確認する
                    if str_list[0] != 'Mat:':
                        continue

                    # コメントアウトされていない場合
                    if str_list[3] == 'ElasticSolid':
                        id = str_list[2]
                        mat = ElasticSolid(float(str_list[5]), float(str_list[7]), float(str_list[9]))
                        material_list.append([id, mat])
                    
                    elif str_list[3] == 'VonMisesSolid':
                        id = str_list[2]
                        mat = ElastoPlasticVonMisesSolid(float(str_list[5]), float(str_list[7]), float(str_list[9]))
                        # 塑性硬化の条件を設定
                        mat.add_stress_plastic_strain_line(400000, 0.0)
                        mat.add_stress_plastic_strain_line(500000, 0.5)
                        mat.add_stress_plastic_strain_line(600000, 0.7)
                        mat.add_stress_plastic_strain_line(700000, 1.0)
                        material_list.append([id, mat])
                    else:
                        print('Error !')

        # inputファイルを閉じる
        input_f.close()

        return material_list

    #---------------------------------------------------------------------
    # 要素情報を作成する
    # 戻り値 elems
    #---------------------------------------------------------------------
    def create_elements(self, connects, commponent_list, material_list):
        
        # コンポーネントのループ
        for id, elem_type, dvalue_type, range in commponent_list:
            
            # コンポーネントが指定する領域が全領域である場合
            if range == 'All':
                
                # 取り扱っているコンポーネントのIDと一致する材料を探索する
                for comp_id, mat in material_list:
                    
                    # 材料におけるコンポーネントIDの確認
                    if id != comp_id:
                        continue

                    # 要素形状の確認
                    if elem_type == 'C3D8_Bbar':

                        # 全要素ループ
                        counter = 0
                        for connect in connects:
                            self.elems.append(C3D8Bbar(counter+1, connect, mat))
                            counter += 1
                        
                    # その他の要素形状
                    else:
                        a = 1

        return self.elems

    #---------------------------------------------------------------------
    # 拘束条件を作成する
    #---------------------------------------------------------------------
    def create_dirichilet_condition(self, nodes):
        
        # 解析条件のinputファイルを開く
        input_f = open(parent_dir + self.input_file_path, 'r')

        # ファイルの読み込みを行う
        while True:

            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = input_f.readline()
            str = str.rstrip('\n')

            # 空白かどうかをチェックし、該当すれば読み取りを終了する
            if str == '':
                break

            # 必要な文字列を取得し、要素を作成する
            if str == '----------------------------------------------------------|Dirichlet_Condition|':
                
                # コメントアウトされていない文字列を探索する
                while True:
                    # 文字列を1行読み込み、末尾の改行'\n'を取り除く
                    str = input_f.readline()
                    str = str.rstrip('\n')
                    
                    # 空白行でループを脱出する
                    if str == '-':
                        break

                    # タブで文字列を分解する
                    str_list = str.split(' ')

                    # コメントアウトされていないか確認する
                    if str_list[0] != 'Analysis:':
                        continue
                    
                    xmin = float(str_list[12]) - 1e-05
                    xmax = float(str_list[14]) + 1e-05
                    ymin = float(str_list[15]) - 1e-05
                    ymax = float(str_list[17]) + 1e-05
                    zmin = float(str_list[18]) - 1e-05
                    zmax = float(str_list[20]) + 1e-05
                    

                    for node in nodes:
                        if xmin < node.x < xmax and ymin < node.y < ymax and zmin < node.z < zmax:
                            
                            flag_x = str_list[4]
                            flag_y = str_list[5]
                            flag_z = str_list[6]
                            
                            if flag_x == '1':
                                val_x = str_list[8]
                            else:
                                val_x = None

                            if flag_y == '1':
                                val_y = str_list[9]
                            else:
                                val_y = None

                            if flag_z == '1':
                                val_z = str_list[10]
                            else:
                                val_z = None

                            self.bound.add_SPC(node.no, float(val_x), float(val_y), float(val_z))
        
        # inputファイルを閉じる
        input_f.close()
        
    #---------------------------------------------------------------------
    # 拘束条件を作成する
    #---------------------------------------------------------------------
    def create_neumman_condition(self, nodes):
        
        # 解析条件のinputファイルを開く
        input_f = open(parent_dir + self.input_file_path, 'r')

        # ファイルの読み込みを行う
        while True:

            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = input_f.readline()
            str = str.rstrip('\n')

            # 空白かどうかをチェックし、該当すれば読み取りを終了する
            if str == '':
                break

            # 必要な文字列を取得し、要素を作成する
            if str == '------------------------------------------------------------|Neumman_Condition|':
                
                # コメントアウトされていない文字列を探索する
                while True:
                    # 文字列を1行読み込み、末尾の改行'\n'を取り除く
                    str = input_f.readline()
                    str = str.rstrip('\n')
                    
                    # 空白行でループを脱出する
                    if str == '-':
                        break

                    # タブで文字列を分解する
                    str_list = str.split(' ')

                    # コメントアウトされていないか確認する
                    if str_list[0] != 'Analysis:':
                        continue
                    
                    xmin = float(str_list[8]) - 1e-05
                    xmax = float(str_list[10]) + 1e-05
                    ymin = float(str_list[11]) - 1e-05
                    ymax = float(str_list[13]) + 1e-05
                    zmin = float(str_list[14]) - 1e-05
                    zmax = float(str_list[16]) + 1e-05
                    
                    for node in nodes:
                        if xmin < node.x < xmax and ymin < node.y < ymax and zmin < node.z < zmax:
                            val_x = str_list[4]
                            val_y = str_list[5]
                            val_z = str_list[6]

                            self.bound.add_force(node.no, float(val_x), float(val_y), float(val_z))
        
        # inputファイルを閉じる
        input_f.close()

    #---------------------------------------------------------------------
    # 実際のモデルを作成する
    #---------------------------------------------------------------------
    def create_model(self):
        
        # 解析メッシュを作成する
        self.nodes, self.connects = self.create_model_mesh()

        # コンポーネントの情報を作成する
        commponent_list = self.create_commponent()

        # 材料モデルの情報を作成する
        material_list = self.create_material_model()

        # 要素情報を作成する
        self.elems = self.create_elements(self.connects, commponent_list, material_list)
        
        # 拘束条件を設定する
        self.bound = Boundary(self.nodes)
        self.create_dirichilet_condition(self.nodes)
        
        # 荷重条件を設定する
        self.create_neumman_condition(self.nodes)

        # 結果を戻す
        return self.nodes, self.elems, self.bound