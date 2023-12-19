from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.command.command import Command
import time
#=============================================================================
# Main 関数
#=============================================================================
def main():

    # 計算の開始時間を取得
    t_start = time.time() 
    
    # commandクラスのオブジェクトを作成し、解析内容をテキストファイルから取得する
    command = Command()
    analysis_list = command.make_analysis(parent_dir + "/input/command")

    # 解析を実行する
    analysis_list[0].run()

    # 計算終了時の時間を取得
    t_end = time.time()

    # 経過時間を表示
    elapsed_time = t_end - t_start
    print(f"経過時間：{elapsed_time}")

#=============================================================================
# Mainを実行し、解析を行う
#=============================================================================
if __name__ == '__main__':
    main()