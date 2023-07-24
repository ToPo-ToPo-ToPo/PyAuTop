
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.analysis.analysis import Analysis
#=============================================================================
#
#=============================================================================
class Command:
    
    def __init__(self):
        self.analysis_list = []
    
    #---------------------------------------------------------------------
    #
    #---------------------------------------------------------------------    
    def make_analysis(self, filepath):
        
        # 解析条件に関するインプットデータを読み込む
        # ファイル:command.datを開く
        command_f = open(filepath + ".dat", 'r')
    
        # ファイルの読み込みを行う
        while True:
            
            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = command_f.readline()
            str.rstrip('\n')

            # 空白かどうかをチェックし、空白であれば読み取りを終了する
            if str == '':
                break
        
            # タブで文字列を分解する
            str_list = str.split(' ')

            # コメントアウトの場合は読み飛ばす
            if str_list[0] != 'ID':
                continue          

            if str_list[3] == 'Analysis':
                # analysisクラスに代入する
                self.analysis_list.append(Analysis(str_list[1], str_list[2], str_list[5], str_list[7], str_list[9], str_list[11]))
            
        # 最後にファイルを閉じる
        command_f.close()

        return self.analysis_list