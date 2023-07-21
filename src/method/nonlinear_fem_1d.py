from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import numpy as np
import numpy.linalg as LA
from src.boundary_1d import Boundary1d

#=============================================================================
#
#=============================================================================
class FEM1d:
    # コンストラクタ
    # nodes    : 節点リスト(節点は1から始まる順番で並んでいる前提(Node1d型のリスト))
    # elements : 要素のリスト(d1t2型のリスト)
    # boundary : 境界条件(d1Boundary型)
    # num_step   : インクリメント数
    def __init__(self, nodes, elements, boundary, num_step):

        self.nodes = nodes         # 節点は1から始まる順番で並んでいる前提(Node1d型のリスト)
        self.elements = elements   # 要素のリスト(d1t2型のリスト)
        self.bound = boundary      # 境界条件(d1Boundary型)
        self.num_step = num_step   # インクリメント数
        self.itr_max = 100         # ニュートン法のイテレータの上限
        self.cn = 0.001            # ニュートン法の変位の収束判定のパラメータ

    #---------------------------------------------------------------------
    # 陰解法で解析を行う
    #---------------------------------------------------------------------
    def analysis(self):

        self.solution_list = []     # インクリメント毎の変位ベクトルのリスト(np.array型のリスト)
        self.Freact_list = []             # インクリメント毎の反力ベクトルのリスト(np.array型のリスト)
        self.elem_output_data_list = []   # インクリメント毎の要素出力のリスト(makeOutputData型のリストのリスト)

        # 荷重をインクリメント毎に分割する
        Fext_list = []
        for i in range(self.num_step):
            Fext_list.append(self.make_force_vector() * (i + 1) / self.num_step)

        # ニュートン法により変位を求める
        solution = np.zeros(len(self.nodes))   # 全節点の変位ベクトル
        R = np.zeros(len(self.nodes))                # 残差力ベクトル
        
        # 増分解析ループ
        for i in range(self.num_step):
            solution_first = solution   # 初期の全節点の変位ベクトル
            Fext = Fext_list[i]                     # i+1番インクリメントの荷重

            # 境界条件を考慮しないインクリメント初期の接線剛性マトリクスKtを作成する
            Kt = self.make_Kt()

            # 境界条件を考慮しないインクリメント初期の残差力ベクトルRを作成する
            if i == 0:
                R = Fext
            else:
                R = Fext - Fext_list[i - 1]

            # 境界条件を考慮したインクリメント初期の接線剛性マトリクスKtc、残差力ベクトルRcを作成する
            Ktc, Rc = self.set_bound_condition(Kt, R)

            # ニュートン法による収束演算を行う
            for j in range(self.itr_max):

                # Ktcの逆行列が計算できるかチェックする
                if np.isclose(LA.det(Ktc), 0.0) :
                    raise ValueError("有限要素法の計算に失敗しました。")

                # 変位増分Δuを計算
                vecd = LA.solve(Ktc, Rc)

                # 変位ベクトルの更新: u_new = u_old + Δu
                solution += vecd

                # 要素内の情報を更新する
                self.update_element_data(solution)

                # 新たな接線剛性マトリクスKtを作成する
                Kt = self.make_Kt()

                # 新たな残差力ベクトルRを求める
                Fint = self.make_Fint()
                R = Fext - Fint

                # 新たな境界条件を考慮したKtcマトリクス、Rcベクトルを作成する
                Ktc, Rc = self.set_bound_condition(Kt, R)

                # 収束判定を行う
                # 増分変位ノルム|Δu|=0であれば収束
                if np.isclose(LA.norm(vecd), 0.0):
                    break

                else:
                    # 増分変位ノルム|Δu|の変化率が微小であれば収束
                    solution_rate = (LA.norm(vecd) / LA.norm(solution - solution_first))
                    if solution_rate < self.cn:
                        break

            # 収束後のインクリメントの変位べクトルを格納する
            self.solution_list.append(solution.copy()) 

            # 収束後の節点反力を計算する
            vecRF = np.array(Fint - Fext).flatten()

            # 収束後のインクリメントの節点反力を格納する
            self.Freact_list.append(vecRF)

    #---------------------------------------------------------------------
    # 結果を整理し、要素情報を更新する
    # solution : 全節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def update_element_data(self, solution):

        # 全要素ループ
        for elem in self.elements:
            
            # 要素elemの変位を初期化
            elem_solution = np.zeros(len(elem.nodes))
            
            # 要素変位の更新
            for i in range(len(elem.nodes)):
                elem_solution[i] = solution[(elem.nodes[i].no - 1)]
            
            # 構成則内の変数を更新
            elem.compute_constitutive_law(elem_solution)        

    #---------------------------------------------------------------------
    # 接線剛性マトリクスKtを作成する
    #---------------------------------------------------------------------
    def make_Kt(self):

        # 初期化
        Kt = np.matrix(np.zeros((len(self.nodes), len(self.nodes))))
        
        # 全要素ループ
        for elem in self.elements:

            # ketマトリクスを計算する
            Ke = elem.make_local_Kt()

            # Ktマトリクスに代入する
            for c in range(len(elem.nodes)):
                
                # 自由度番号の取得
                ct = (elem.nodes[c].no - 1)
                
                for r in range(len(elem.nodes)):

                    # 自由度番号の取得
                    rt = (elem.nodes[r].no - 1)

                    # アセンブリング
                    Kt[ct, rt] += Ke[c, r]

        return Kt
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    def make_Fint(self):

        # 初期化
        Fint = np.zeros(len(self.nodes))
        
        # 全体要素ループ
        for elem in self.elements:
            
            # 要素内力ベクトルFint_eの作成
            Fe = elem.make_local_Fint()
            
            # アセンブリングによる全体内力ベクトルFintを作成
            for k in range(len(elem.nodes)):
                Fint[(elem.nodes[k].no - 1)] += Fe[k]
        
        return Fint

    #---------------------------------------------------------------------
    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_force_vector(self):

        # 節点に負荷する荷重ベクトルを作成する
        vecf = self.bound.make_force_vector()

        return vecf

    #---------------------------------------------------------------------
    # 接線剛性マトリクス、残差力ベクトルに境界条件を考慮する
    #---------------------------------------------------------------------
    def set_bound_condition(self, Kt, R):

        # 初期化
        Ktc = np.copy(Kt)
        Rc = np.copy(R)

        # 強制変位ベクトルを取得
        solution = self.bound.make_disp_vector()

        # 単点拘束条件を考慮した接線剛性マトリクス、残差力ベクトルを作成する
        for idof in range(len(solution)):    
            if not solution[idof] == None:

                # Ktマトリクスからi列を抽出する
                vecx = np.array(Kt[:, idof]).flatten()

                # 変位ベクトルi列の影響を荷重ベクトルに適用する
                Rc = Rc - solution[idof] * vecx

                # Ktマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                Ktc[:, idof] = 0.0
                Ktc[idof, :] = 0.0
                Ktc[idof, idof] = 1.0
        
        # 強制変位量を方程式の右辺へ格納する
        for idof in range(len(solution)):
            if not solution[idof] == None:
                Rc[idof] = solution[idof]

        return Ktc, Rc

    #---------------------------------------------------------------------
    # 解析結果をテキストファイルに出力する
    #---------------------------------------------------------------------
    def output_txt(self, filePath):

        # ファイルを作成し、開く
        f = open(filePath + ".txt", 'w')

        # 出力する文字の情報を定義する
        columNum = 20
        floatDigits = ".10g"

        # 入力データのタイトルを書きこむ
        f.write("*********************************\n")
        f.write("*          Input Data           *\n")
        f.write("*********************************\n")
        f.write("\n")

        # 節点情報を出力する
        f.write("***** Node Data ******\n")
        f.write("No".rjust(columNum) + "X".rjust(columNum) + "\n")
        f.write("-" * columNum * 2 + "\n")
        for node in self.nodes:
            strNo = str(node.no).rjust(columNum)
            strX = str(format(node.x, floatDigits).rjust(columNum))
            f.write(strNo + strX + "\n")
        f.write("\n")

        # 要素情報を出力する
        '''nodeNoColumNum = columNum
        f.write("***** Element Data ******\n")
        f.write("No".rjust(columNum) + "Type".rjust(columNum) + "Node No".rjust(nodeNoColumNum) + 
                "Young".rjust(columNum) + "Area".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "-" * nodeNoColumNum + "\n")
        for elem in self.elements:
            strNo = str(elem.no).rjust(columNum)
            strType = str(elem.__class__.__name__ ).rjust(columNum)
            strNodeNo = ""
            for node in elem.nodes:
                strNodeNo += " " + str(node.no)
            strNodeNo = strNodeNo.rjust(nodeNoColumNum)
            strYoung = str(format(elem.mat.young, floatDigits).rjust(columNum))
            strArea = str(format(elem.area, floatDigits).rjust(columNum))
            f.write(strNo + strType + strNodeNo + strYoung + strArea + "\n")
        f.write("\n")'''

        # 単点拘束情報を出力する
        '''f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "\n")
        f.write("-" * columNum * 2 + "\n")
        for i in range(len(self.bound.dispNodeNo)):
            strNo = str(self.bound.dispNodeNo[i]).rjust(columNum)
            strXDisp = "None".rjust(columNum)
            if not self.bound.dispX[i] is None:
                strXDisp = str(format(self.bound.dispX[i], floatDigits).rjust(columNum))
            f.write(strNo + strXDisp + "\n")
        f.write("\n")'''

        # 荷重条件を出力する(等価節点力も含む)
        f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "\n")
        f.write("-" * columNum * 2 + "\n")
        vecf = self.make_force_vector()
        for i in range(len(self.nodes)):
            if not vecf[i] == 0 :
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[i], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + "\n")
        f.write("\n")

        # 結果データのタイトルを書きこむ
        f.write("**********************************\n")
        f.write("*          Result Data           *\n")
        f.write("**********************************\n")
        f.write("\n")

        for i in range(self.num_step):
            f.write("*Increment " + str(i + 1) + "\n")
            f.write("\n")

            # 変位のデータを出力する
            f.write("***** Displacement Data ******\n")
            f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "\n")
            f.write("-" * columNum * 2 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                solution = self.solution_list[i]
                strXDisp = str(format(solution[j], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + "\n")            
            f.write("\n")

            # 反力のデータを出力する
            f.write("***** Reaction Force Data ******\n")
            f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "\n")
            f.write("-" * columNum * 2 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecRF = self.Freact_list[i]
                strXForce = str(format(vecRF[j], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + "\n")            
            f.write("\n")

        # ファイルを閉じる
        f.close()
