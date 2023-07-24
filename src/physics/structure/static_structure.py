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
                    xdiv = float(str_list[9])
                    ydiv = float(str_list[11])
                    zdiv = float(str_list[13])

                    #self.nodes, self.connects = auto_mesh(mesh_type, xlength, ylength, zlength, xdiv, ydiv, zdiv)
        
        # inputファイルを閉じる
        input_f.close()

        # 解析メッシュを生成する
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
        self.nodes = [node1, node2, node3, node4, node5, node6, node7, node8,
                      node9, node10, node11, node12, node13, node14, node15, node16]
        
        # 要素を構成する節点のコネクティビティを設定する
        nodes1 = [node1, node2, node10, node9, node5, node6, node14, node13]
        nodes2 = [node2, node3, node11, node10, node6, node7, node15, node14]
        nodes3 = [node3, node4, node12, node11, node7, node8, node16, node15]
        self.connects = [nodes1, nodes2, nodes3]

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