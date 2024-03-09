
from linear_elastic_plane_stress import LinearElasticPlaneStress
from linear_elastic_brick import LinearElasticBrick
import make_hexa8_voxel_model
#=============================================================================
# メインプログラム
#=============================================================================
def main_test():
    
    # 節点とコネクティビティを作成
    nodes, elements = make_hexa8_voxel_model.run(xlength=3.0, ylength=1.0, zlength=1.0, xdiv=30, ydiv=10, zdiv=10)
    
    # 材料物性を定義
    material = LinearElasticBrick(young=1.0, poisson=0.3, density=1.0e-05)
    for element in elements:
        element.set_material(material)
        
    
    # 境界条件の設定
    
    
    # 解析モデルの作成
    
    
    # 解析手法の設定
    
    
    # 解析の実行
    
    
    # ポスト処理
    
    
#=============================================================================
# メインプログラム
#=============================================================================
main_test()