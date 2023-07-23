from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.command.command import Command
from src.physics.structure.static_structure import StaticStructure
#=============================================================================
# Main 関数
#=============================================================================
def main():
    
    # commandクラスのオブジェクトを作成し、解析内容をテキストファイルから取得する
    command = Command()
    mesh_type_list, physics_list, method_list, analysis_list = command.run()
    
    # 解析のmainを作成する
    for ianalysis in range(len(analysis_list)):
        
        # 解析で取り扱う物理モデルの番号を取得する
        physics_id = analysis_list[ianalysis][1]
        
        # 番号に一致するものを探す
        for iphysics in range(len(physics_list)):
            if physics_id == physics_list[iphysics][0]:
                
                # オブジェクトを生成する
                if physics_list[iphysics][1] == 'Static_Structure':
                    physics = StaticStructure(physics_id)
                
                # 見つけた場合はループから抜ける
                break
        
        # 解析で使用する手法を作成する
        method_id = analysis_list[ianalysis][2]
        
        # 番号に一致しているものを探す
        for imethod in range(len(method_list_list)):
            if method_id == method_list[imethod][0]:
                
                # オブジェクトを生成する
                if method_list[imethod][1] == 'Nonlinear_FEM':
                    method = StaticStructure(physics_id)
                
                # 見つけた場合はループから抜ける
                break
    
    # メッシュを生成する
    

#=============================================================================
# Mainを実行し、解析を行う
#=============================================================================
if __name__ == '__main__':
    main()