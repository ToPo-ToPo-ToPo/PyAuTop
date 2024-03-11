
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from solid_mechanics import SolidMechanics
#=============================================================================
# 線形FEM解析のクラス
#=============================================================================
class LinearFEM:
    #---------------------------------------------------------------------
    # コンストラクタ
    #---------------------------------------------------------------------
    def __init__(self, model, num_step):
        # インスタンス変数を定義する
        self.model = model
        self.num_step = num_step 
        
    #---------------------------------------------------------------------
    # 解析を行う
    #---------------------------------------------------------------------
    #@partial(jit, static_argnums=(0))
    def run(self):
        
        # 変位ベクトルのデータ
        U_list = jnp.array(jnp.zeros((self.num_step+1, self.model.num_total_equation)))
        U = jnp.zeros(self.model.num_total_equation)
        
        # 反力に関するデータ
        Freact_list = jnp.array(jnp.zeros((self.num_step+1, self.model.num_total_equation)))

        # 境界条件を考慮しないKマトリクスを作成
        K = self.model.make_K()

        # 増分解析ループ
        for istep in range(self.num_step+1):
            
            # 計算中の情報を出力
            print('')
            print('============================================================')
            print(' Incremental step' + str(istep+1))
            print('------------------------------------------------------------')
            print(' ||Fext||                ||u||                              ')
            print('------------------------------------------------------------')

            # i番インクリメントの荷重を設定する
            Fext = self.model.make_Ft(istep)
            #print(Fext)

            # 境界条件を考慮したKマトリクス、荷重ベクトルを作成する
            lhs_c, rhs_c = self.model.consider_dirichlet_bc(istep, K, Fext, U)
            #print(lhs_c)

            # 変位ベクトルを計算し、インクリメントの最終的な変位べクトルを格納する
            U = jax.scipy.linalg.solve(lhs_c, rhs_c, check_finite=False)
            #U_list[istep, :] = U.copy()
            U_list = U_list.at[istep, :].set(U.copy())

            # 節点反力を計算し、インクリメントの最終的な節点反力を格納する
            Freact = jnp.array(K @ U - Fext).flatten()
            #Freact_list[istep, :] = Freact.copy()
            Freact_list = Freact_list.at[istep, :].set(Freact.copy())
            
        return U_list, Freact_list

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
        '''nodeNoColumNum = 36
        f.write("***** Element Data ******\n")
        f.write("No".rjust(columNum) + "Type".rjust(columNum) + "Node No".rjust(nodeNoColumNum) + 
                "Young".rjust(columNum) + "Poisson".rjust(columNum) + "Density".rjust(columNum) + "\n")
        f.write("-" * columNum * 5 + "-" * nodeNoColumNum + "\n")
        for elem in self.elements:
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
            strDensity = "None".rjust(columNum)
            if not elem.density is None:
                strDensity = str(format(elem.density, floatDigits).rjust(columNum))
            f.write(strNo + strType + strNodeNo + strYoung + strPoisson + strDensity + "\n")
        f.write("\n")'''

        # 単点拘束情報を出力する
        f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) +"\n")
        f.write("-" * columNum * 4 + "\n")
        vecd = self.bound.make_disp_vector()
        for i in range(len(self.nodes)):
            flg = False
            for j in range(self.nodes[i].num_dof):
                if not vecd[self.nodes[i].num_dof * i + j] == None:
                    flg = True
            if flg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = str(format(vecd[self.nodes[i].num_dof * i], floatDigits).rjust(columNum))
                strYDisp = str(format(vecd[self.nodes[i].num_dof * i + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(vecd[self.nodes[i].num_dof * i + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")
        f.write("\n")

        # 荷重条件を出力する(等価節点力も含む)
        f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + "Z Force".rjust(columNum) +"\n")
        f.write("-" * columNum * 4 + "\n")
        vecf = self.make_Fext()
        for i in range(len(self.nodes)):
            flg = False
            for j in range(self.nodes[i].num_dof):
                if not vecf[self.nodes[i].num_dof * i + j] == None:
                    flg = True
            if flg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[self.nodes[i].num_dof * i], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[self.nodes[i].num_dof * i + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[self.nodes[i].num_dof * i + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
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
