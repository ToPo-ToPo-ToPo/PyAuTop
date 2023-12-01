
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.physics.structure.static_structure import StaticStructure
from src.method.linear_fem import LinearFEM
from src.method.nonlinear_fem import NonlinearFEM
from src.method.nonlinear_fem_2d import NonlinearFEM2d
from src.physics.node import Node
from src.material.elasto_plastic_von_mises.solid import ElastoPlasticVonMisesSolid
from src.boundary import Boundary
from src.physics.element.C3D8_Bbar import C3D8Bbar
#=============================================================================
#
#=============================================================================
class Analysis:
    
    def __init__(self, id, analysis_type, mesh_type, physics_type, method_type, num_step):
        
        # 基本的な値の初期化
        self.id = id
        self.analysis_type = analysis_type

        # 解析に必要なインスタンスを作成する
        nodes = []
        elems = []
        bound = []

        # Automesh用のinputを使用する場合
        if mesh_type == 'Auto':

            # 物理モデルphysicsのオブジェクトを生成する
            if physics_type == 'Static_Structure':
     
                # オブジェクトを作成する
                self.physics = StaticStructure(id)

                # 解析モデルを作成する
                nodes, elems, bound = self.physics.create_model()
            
            else:
                a = 1
            
            # 解析クラスmethodのオブジェクトを作成する
            # 線形解析の有限要素法を使用する
            if method_type == 'Linear_FEM':
                self.method = LinearFEM(nodes, elems, bound, int(num_step))
            
             # 非線形解析の有限要素法を使用する
            elif method_type == 'Nonlinear_FEM':
                self.method = NonlinearFEM(nodes, elems, bound, int(num_step))
                
             # 非線形解析の有限要素法を使用する
            elif method_type == 'Nonlinear_FEM_2D':
                self.method = NonlinearFEM2d(nodes, elems, bound, int(num_step))
            
            else:
                a = 1
        
        # その他のinputファイルを使用する場合
        else:
            a = 1
    
    #---------------------------------------------------------------------
    # 解析を実行する
    #---------------------------------------------------------------------
    def run(self):
        
        #for istep in range(self.num_step):
        self.method.run()

        # 結果を出力する
        # self.method.output_txt(parent_dir +  "/output/C3D8_test_r1")
        self.method.output_vtk(parent_dir +  "/output/CPS4_test")