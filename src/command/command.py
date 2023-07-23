

class Command:
    
    def __init__(self):
        self.file_name = '../../input/command.dat'
    
    #---------------------------------------------------------------------
    #
    #---------------------------------------------------------------------    
    def make_analysis(self):
        
        # 解析条件に関するインプットデータを読み込む
        # ファイル:command.datを開く
        command_f = open(self.file_name, 'r')
    
        # ファイルの読み込みを行う
        while True:
            
            # 文字列を1行読み込み、末尾の改行'\n'を取り除く
            str = command_f.readline()
            str.rstrip('\n')
        
            # メッシュ作成方法を定義する
            if str == '--------------------------------------------------------------|Modeling_Method|':
            
                # 文字列を1行読み込み、末尾の改行'\n'を除去する
                str_in = command_f.readline()
                str_in.rstrip('\n')
                str_in.split('\t')
                
                # 空白かどうかをチェックし、空白であれば戻る
                if str_in == '':
                    continue
                
                # タブで文字列を分解する
                str_in.split('\t')
                
                # keywordをチェック
                if str_in[0] == 'Auto_Mesh':
                    self.mesh_type_list.append('Auto_Mesh')
                    
            # 取り扱う物理モデルを定義する
            elif str == '----------------------------------------------------------------------|Physics|':
                
                # 文字列を1行読み込み、末尾の改行'\n'を除去する
                str_in = command_f.readline()
                str_in.rstrip('\n')
                str_in.split('\t')
                
                # 空白かどうかをチェックし、空白であれば戻る
                if str_in == '':
                    continue
                
                # アクティブとなっているかチェックし、データを保存する
                if str_in[0] == 'ID':
                    id = str_in[1]
                    if str_in[2] == 'Static_Structure':
                        self.physics_list.append(id, 'Static_Structure')
        
            # 方程式の解法を定義する
            elif str == '-----------------------------------------------------------------------|Method|':
                # 文字列を1行読み込み、末尾の改行'\n'を除去する
                str_in = command_f.readline()
                str_in.rstrip('\n')
                str_in.split('\t')
                
                # 空白かどうかをチェックし、空白であれば戻る
                if str_in == '':
                    continue
                
                # アクティブとなっているかチェックし、データを保存する
                if str_in[0] == 'ID':
                    id = str_in[1]
                    if str_in[2] == 'Linear_FEM':
                        self.method_list.append(id, 'Linear_FEM')
                    elif str_in[2] == 'Nonlinear_FEM':
                        self.method_list.append(id, 'Nonlinear_FEM')
                        
            # 解析モデルを作成する
            elif str == '---------------------------------------------------------------------|Analysis|':
                # 文字列を1行読み込み、末尾の改行'\n'を除去する
                str_in = command_f.readline()
                str_in.rstrip('\n')
                str_in.split('\t')
                
                # 空白かどうかをチェックし、空白であれば戻る
                if str_in == '':
                    continue
                
                # アクティブとなっているかチェックし、データを保存する
                if str_in[0] == 'ID':
                    id = str_in[1]
                    type = str_in[2]
                    
                    # 解析の種類を確認する
                    if str_in[3] == 'Analysis':
                        
                        # 各種番号をストックする
                        physics_id = str_in[5]
                        method_id = str_in[7]
                        num_step = str_in[9]
                        
                        # リストを作成する
                        self.analysis_list.append(id, type, 'Analysis', physics_id, method_id, num_step)
                                
            # ファイル読み込み終了の判定を行う
            elif str == '--------------------------------------------------------------------------|End|':
                break
    
        # 最後にファイルを閉じる
        command_f.close()
        
        # 作成したリストを返す
        return self.mesh_type_list, self.physics_list, self.method_list, self.analysis_list