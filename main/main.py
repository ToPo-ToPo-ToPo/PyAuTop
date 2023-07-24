from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.command.command import Command
#=============================================================================
# Main 関数
#=============================================================================
def main():
    
    # commandクラスのオブジェクトを作成し、解析内容をテキストファイルから取得する
    command = Command()
    analysis_list = command.make_analysis(parent_dir + "/input/command")

    # 解析を実行する
    analysis_list[0].run()
    

#=============================================================================
# Mainを実行し、解析を行う
#=============================================================================
if __name__ == '__main__':
    main()