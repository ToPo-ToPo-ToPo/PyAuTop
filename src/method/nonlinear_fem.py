#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from src.method.fem_base import FEMBase

#=============================================================================
#
#=============================================================================
class NonlinearFEM(FEMBase):
    # コンストラクタ
    # nodes    : 節点は1から始まる順番で並んでいる前提(Node型のリスト)
    # elements : 要素は種類ごとにソートされている前提(C3D８型のリスト)
    # bound    : 境界条件(d2Boundary型)
    # num_step   : インクリメント数
    def __init__(self, nodes, elements, bound, num_step):

        # インスタンス変数を定義する
        self.nodes = nodes                 # 節点は1から始まる順番で並んでいる前提(Node2d型のリスト)
        self.elements = elements           # 要素は種類ごとにソートされている前提(リスト)
        self.bound = bound                 # 境界条件(d2Boundary型)
        
        self.num_step = num_step           # インクリメント数
        self.itr_max = 20                  # ニュートン法のイテレータの上限
        
        self.rn = 1.0e-07                  # ニュートン法の残差力の収束判定のパラメータ
        self.cn = 1.0e-06                  # ニュートン法の変位の収束判定のパラメータ

        # 総自由度数を計算する
        self.compute_num_total_equation()
        
    #---------------------------------------------------------------------
    # 陰解法で解析を行う
    #---------------------------------------------------------------------
    def run(self):

        self.solution_list = []   # インクリメント毎の変位ベクトルのリスト(np.array型のリスト)
        self.Freact_list = []     # インクリメント毎の反力ベクトルのリスト(np.array型のリスト)

        # 荷重をインクリメント毎に分割する
        Fext_list = []
        for istep in range(self.num_step):
            Fext_list.append(self.make_Fext() * (istep + 1) / self.num_step)
        
        # Dirichiret境界の規定値をインクリメント毎に分割する
        solution_bar_list = []
        for istep in range(self.num_step):

            # 初期化
            solution_bar_tmp = self.bound.make_disp_vector().copy()

            # 境界条件を設定した節点のみを対象にする
            for i in range(len(solution_bar_tmp)):
                if not solution_bar_tmp[i] == None:
                    solution_bar_tmp[i] *= (istep + 1) / self.num_step
            
            # 保存
            solution_bar_list.append(solution_bar_tmp.copy())


        # 変位ベクトルと残差ベクトルの定義
        solution = np.zeros(self.num_total_equation)   # 全節点の変位ベクトル
        R = np.zeros(self.num_total_equation)          # 残差力ベクトル
        
        # 増分解析ループ
        for istep in range(self.num_step):

            # 計算中の情報を出力
            print('')
            print('============================================================')
            print(' Incremental step' + str(istep+1))
            print('')
            print('------------------------------------------------------------')
            print(' Iter     ||R0||         ||R/R0||       ||δu/Δu||           ')
            print('------------------------------------------------------------')

            # 初期化
            solution_first = solution.copy()                   # 初期の全節点の変位ベクトル
            Fext = Fext_list[istep]                            # istep番インクリメントの荷重
            solution_bar = solution_bar_list[istep]            # istep番目のdirichlet境界の規定値

            # 接線剛性マトリクスKtを作成する
            Kt = self.make_K()

            # 境界条件を考慮しないインクリメント初期の残差力ベクトルRを作成する
            if istep == 0:
                R = Fext_list[0]
            else:
                R = Fext_list[istep] - Fext_list[istep - 1]

            # 境界条件を考慮したKtcマトリクス、Rcベクトルを作成する
            Ktc, Rc = self.set_bound_condition(Kt, R, solution_bar, solution)

            # 初期の残差ノルムを計算する
            residual0 = LA.norm(Rc)

            # ニュートン法による収束演算を行う
            for iter in range(self.itr_max + 1):

                # 疎行列に変換する
                Ktc = csr_matrix(Ktc)

                # 変位増分Δuを計算
                delta_solution = spsolve(Ktc, Rc, use_umfpack=True)

                # 変位ベクトルの更新: u_new = u_old + Δu
                solution += delta_solution

                # 全ての要素内の変数を更新する
                self.update_element_data(solution)

                # 新たな接線剛性マトリクスKtを作成する
                Kt = self.make_K()

                # 新たな残差力ベクトルRを求める
                Fint = self.make_Fint()
                R = Fext - Fint

                # 新たな境界条件を考慮したKtcマトリクス、Rcベクトルを作成する
                Ktc, Rc = self.set_bound_condition(Kt, R, solution_bar, solution)

                # 収束判定に必要な変数を計算する
                check_flag, solution_rate, residual_rate = self.check_convergence(Fint, Rc, solution, solution_first, delta_solution)

                # 計算中の情報を出力
                print(' ' + str(iter+1) + '      ' 
                      + '{:.4e}'.format(residual0) + '      ' 
                      + '{:.4e}'.format(residual_rate) + '      ' 
                      + '{:.4e}'.format(solution_rate))

                # 収束判定を行う
                if check_flag == True:
                    break
                if iter + 1 >= self.itr_max:
                     raise ValueError("ニュートンラプソン法が収束しませんでした。")

            # 構成則の情報の更新
            self.update_constitutive_low()

            # インクリメントの最終的な変位べクトルを格納する
            self.solution_list.append(solution.copy())

            # 節点反力を計算する
            # つり合っている場合、FintとFextの違いは反力のみとなる
            Fint = self.make_Fint()
            Rreact = np.array(Fint - Fext).flatten()

            # インクリメントの最終的な節点反力を格納する
            self.Freact_list.append(Rreact)      

    #---------------------------------------------------------------------
    # ニュートンラプソン法の収束判定を行う
    #---------------------------------------------------------------------
    def check_convergence(self, Fint, Rc, solution, solution_first, delta_solution):
        
        # 初期化
        check_flag = False

        # 残差ベクトルの全成分または変位増分がゼロの場合、収束とする
        if LA.norm(delta_solution) < self.cn * 1.0e-03 and LA.norm(Rc) < self.rn * 1.0e-03:
            check_flag = True

        # 増分変位における相対誤差を計算する
        solution_rate = LA.norm(delta_solution) / LA.norm(solution - solution_first)
        
        # 残差ベクトルにおける相対誤差を計算する
        residual_rate = LA.norm(Rc) / LA.norm(Fint)

        # 増分変位と残差ベクトルが閾値以下であれば、収束とする
        if solution_rate < self.cn and residual_rate < self.rn:
            check_flag = True
        
        return check_flag, solution_rate, residual_rate

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
        f.write("No".rjust(columNum) + "X".rjust(columNum) + "Y".rjust(columNum) + "Z".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        for node in self.nodes:
            strNo = str(node.no).rjust(columNum)
            strX = str(format(node.x, floatDigits).rjust(columNum))
            strY = str(format(node.y, floatDigits).rjust(columNum))
            strZ = str(format(node.z, floatDigits).rjust(columNum))
            f.write(strNo + strX + strY + strZ + "\n")
        f.write("\n")

        # 要素情報を出力する
        nodeNoColumNum = 36
        f.write("***** Element Data ******\n")
        f.write("No".rjust(columNum) + "Type".rjust(columNum) + "Node No".rjust(nodeNoColumNum) + 
                "Young".rjust(columNum) + "Poisson".rjust(columNum) + "Thickness".rjust(columNum) + 
                "Area".rjust(columNum) + "Density".rjust(columNum) + "\n")
        f.write("-" * columNum * 7 + "-" * nodeNoColumNum + "\n")
        """for elem in self.elements:
            strNo = str(elem.no).rjust(columNum)
            strType = str(elem.__class__.__name__ ).rjust(columNum)
            strNodeNo = ""
            for node in elem.nodes:
                strNodeNo += " " + str(node.no)
            strNodeNo = strNodeNo.rjust(nodeNoColumNum)
            strYoung = str(format(elem.young, floatDigits).rjust(columNum))
            strPoisson = "None".rjust(columNum)
            if hasattr(elem, 'poisson'):
                strPoisson = str(format(elem.poisson, floatDigits).rjust(columNum))
            strThickness = "None".rjust(columNum)
            if hasattr(elem, 'thickness'):
                strThickness = str(format(elem.thickness, floatDigits).rjust(columNum))
            strArea = "None".rjust(columNum)
            if hasattr(elem, 'area'):
                strArea = str(format(elem.area, floatDigits).rjust(columNum))
            strDensity = "None".rjust(columNum)
            if not elem.density is None:
                strDensity = str(format(elem.density, floatDigits).rjust(columNum))
            f.write(strNo + strType + strNodeNo + strYoung + strPoisson + 
                    strThickness + strArea + strDensity + "\n")"""
        f.write("\n")

        # 単点拘束情報を出力する
        '''f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "Y Displacement".rjust(columNum) + 
                "Z Displacement".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecBoundDisp = self.bound.make_disp_vector()
        for i in range(self.bound.num_node):
            strFlg = False
            for j in range(self.bound.num_dof_at_node):
                if not vecBoundDisp[i * self.bound.num_dof_at_node + j] is None:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.num_dof_at_node] is None:
                    strXDisp = str(format(vecBoundDisp[i * self.bound.num_dof_at_node], floatDigits).rjust(columNum))
                strYDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.num_dof_at_node + 1] is None:
                    strYDisp = str(format(vecBoundDisp[i * self.bound.num_dof_at_node + 1], floatDigits).rjust(columNum))
                strZDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.num_dof_at_node + 2] is None:
                    strZDisp = str(format(vecBoundDisp[i * self.bound.num_dof_at_node + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")
        f.write("\n")'''

        # 荷重条件を出力する(等価節点力も含む)
        """f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                "Z Force".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecf = self.make_Fext()
        for i in range(len(self.nodes)):
            strFlg = False
            for j in range(self.bound.num_dof_at_node):
                if not vecf[i * self.bound.num_dof_at_node + j] == 0.0:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[i * self.bound.num_dof_at_node], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[i * self.bound.num_dof_at_node + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[i * self.bound.num_dof_at_node + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
        f.write("\n")"""

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
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Displacement".rjust(columNum) +
                    "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                solution = self.solution_list[i]
                mag = np.linalg.norm(np.array((solution[self.nodes[j].num_dof * j], solution[self.nodes[j].num_dof * j + 1], solution[self.nodes[j].num_dof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXDisp = str(format(solution[self.nodes[j].num_dof * j], floatDigits).rjust(columNum))
                strYDisp = str(format(solution[self.nodes[j].num_dof * j + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(solution[self.nodes[j].num_dof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")            
            f.write("\n")

            # 応力データを出力する
            """f.write("***** Stress Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "Stress XX".rjust(columNum) + "Stress YY".rjust(columNum) + 
                    "Stress ZZ".rjust(columNum) + "Stress XY".rjust(columNum) + "Stress XZ".rjust(columNum) + "Stress YZ".rjust(columNum) +
                    "Mises".rjust(columNum) + "\n")
            f.write("-" * columNum * 9 + "\n")
            for elemOutputData in self.elem_output_data_list[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strStressXX = str(format(elemOutputData.vecIpStressList[j][0], floatDigits).rjust(columNum))
                    strStressYY = str(format(elemOutputData.vecIpStressList[j][1], floatDigits).rjust(columNum))
                    strStressZZ = str(format(elemOutputData.vecIpStressList[j][2], floatDigits).rjust(columNum))
                    strStressXY = str(format(elemOutputData.vecIpStressList[j][5], floatDigits).rjust(columNum))
                    strStressXZ = str(format(elemOutputData.vecIpStressList[j][4], floatDigits).rjust(columNum))
                    strStressYZ = str(format(elemOutputData.vecIpStressList[j][3], floatDigits).rjust(columNum))
                    strMises = str(format(elemOutputData.ipMiseses[j], floatDigits).rjust(columNum))
                    f.write(strElemNo + strIntNo + strStressXX + strStressYY + strStressZZ + 
                            strStressXY + strStressXZ + strStressYZ + strMises + "\n")
            f.write("\n")

            # 全ひずみデータを出力する
            f.write("***** Strain Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "Strain XX".rjust(columNum) + "Strain YY".rjust(columNum) + 
                    "Strain ZZ".rjust(columNum) + "Strain XY".rjust(columNum) + "Strain XZ".rjust(columNum) + "Strain YZ".rjust(columNum) + "\n")
            f.write("-" * columNum * 8 + "\n")
            for elemOutputData in self.elem_output_data_list[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strStrainXX = str(format(elemOutputData.vecIpStrainList[j][0], floatDigits).rjust(columNum))
                    strStrainYY = str(format(elemOutputData.vecIpStrainList[j][1], floatDigits).rjust(columNum))
                    strStrainZZ = str(format(elemOutputData.vecIpStrainList[j][2], floatDigits).rjust(columNum))
                    strStrainXY = str(format(elemOutputData.vecIpStrainList[j][5], floatDigits).rjust(columNum))
                    strStrainXZ = str(format(elemOutputData.vecIpStrainList[j][4], floatDigits).rjust(columNum))
                    strStrainYZ = str(format(elemOutputData.vecIpStrainList[j][3], floatDigits).rjust(columNum))
                    f.write(strElemNo + strIntNo + strStrainXX + strStrainYY + strStrainZZ + 
                            strStrainXY + strStrainXZ + strStrainYZ +"\n")
            f.write("\n")

            # 塑性ひずみデータを出力する
            f.write("***** Plastic Strain Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "PStrain XX".rjust(columNum) + "PStrain YY".rjust(columNum) + 
                    "PStrain ZZ".rjust(columNum) + "PStrain XY".rjust(columNum) + "PStrain XZ".rjust(columNum) + "PStrain YZ".rjust(columNum) + 
                    "Equivalent Plastic Strain".rjust(30) + "\n")
            f.write("-" * (columNum * 8 + 30) + "\n")
            for elemOutputData in self.elem_output_data_list[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strPStrainXX = str(format(elemOutputData.vecIpPStrainList[j][0], floatDigits).rjust(columNum))
                    strPStrainYY = str(format(elemOutputData.vecIpPStrainList[j][1], floatDigits).rjust(columNum))
                    strPStrainZZ = str(format(elemOutputData.vecIpPStrainList[j][2], floatDigits).rjust(columNum))
                    strPStrainXY = str(format(elemOutputData.vecIpPStrainList[j][5], floatDigits).rjust(columNum))
                    strPStrainXZ = str(format(elemOutputData.vecIpPStrainList[j][4], floatDigits).rjust(columNum))
                    strPStrainYZ = str(format(elemOutputData.vecIpPStrainList[j][3], floatDigits).rjust(columNum))                    
                    strEPStrain = str(format(elemOutputData.ipEPStrainList[j], floatDigits).rjust(30))
                    f.write(strElemNo + strIntNo + strPStrainXX + strPStrainYY + strPStrainZZ + 
                            strPStrainXY + strPStrainXZ + strPStrainYZ + strEPStrain + "\n")
            f.write("\n") """          

            # 反力のデータを出力する
            f.write("***** Reaction Force Data ******\n")
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                    "Z Force".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecRF = self.Freact_list[i]
                mag = np.linalg.norm(np.array((vecRF[self.nodes[j].num_dof * j], vecRF[self.nodes[j].num_dof * j + 1], vecRF[self.nodes[j].num_dof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXForce = str(format(vecRF[self.nodes[j].num_dof * j], floatDigits).rjust(columNum))
                strYForce = str(format(vecRF[self.nodes[j].num_dof * j + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecRF[self.nodes[j].num_dof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")            
            f.write("\n")

        # ファイルを閉じる
        f.close()
