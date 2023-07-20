import numpy as np
import numpy.linalg as LA
from boundary import Boundary

class LinearFEM:
    # コンストラクタ
    # nodes    : 節点は1から始まる順番で並んでいる前提(Node型のリスト)
    # elements : 要素は種類ごとにソートされている前提(C3D4型のリスト)
    # bound    : 境界条件(d2Boundary型)
    def __init__(self, nodes, elements, bound):

        # インスタンス変数を定義する
        self.num_dof_at_node = 3            # 節点の自由度
        self.nodes = nodes          # 節点のリスト
        self.elements = elements    # 要素のリスト
        self.bound = bound          # 境界条件

    # 解析を行う
    def analysis(self):

        # 境界条件を考慮しないKマトリクスを作成する
        matK = self.makeKmatrix()

        # 荷重ベクトルを作成する
        vecf = self.make_force_vector()

        # 境界条件を考慮したKマトリクス、荷重ベクトルを作成する
        matKc, vecfc = self.set_bound_condition(matK, vecf)

        if np.isclose(LA.det(matKc), 0.0) :
            raise ValueError("有限要素法の計算に失敗しました。")

        # 変位ベクトルを計算する
        physical_field = LA.solve(matKc, vecfc)
        self.physical_field = physical_field

        # 節点反力を計算する
        vecRF = np.array(matK @ physical_field - vecf).flatten()
        self.vecRF = vecRF

        return physical_field, vecRF

    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    def make_force_vector(self):

        # 節点に負荷する荷重ベクトルを作成する
        vecCondiForce = self.bound.make_force_vector()

        # 等価節点力の荷重ベクトルを作成する
        vecEqNodeForce = np.zeros(len(self.nodes) * self.num_dof_at_node)
        for elem in self.elements:
            vecElemEqNodeForce = elem.makeEqNodeForceVector()
            for i in range(len(elem.nodes)):
                for j in range(self.num_dof_at_node):
                    vecEqNodeForce[self.num_dof_at_node * (elem.nodes[i].no - 1) + j] += vecElemEqNodeForce[self.num_dof_at_node * i + j]

        # 境界条件、等価節点力の荷重ベクトルを足し合わせる
        vecf = vecCondiForce + vecEqNodeForce

        return vecf

    # 境界条件を考慮しないKマトリクスを作成する
    def makeKmatrix(self):

        matK = np.matrix(np.zeros((len(self.nodes) * self.num_dof_at_node, len(self.nodes) * self.num_dof_at_node)))
        for elem in self.elements:

            # ketマトリクスを計算する
            matKe = elem.makeKematrix()

            # Ktマトリクスに代入する
            for c in range(len(elem.nodes) * self.num_dof_at_node):
                ct = (elem.nodes[c // self.num_dof_at_node].no - 1) * self.num_dof_at_node + c % self.num_dof_at_node
                for r in range(len(elem.nodes) * self.num_dof_at_node):
                    rt = (elem.nodes[r // self.num_dof_at_node].no - 1) * self.num_dof_at_node + r % self.num_dof_at_node
                    matK[ct, rt] += matKe[c, r]

        return matK

    # Kマトリクス、荷重ベクトルに境界条件を考慮する
    # matK         : 剛性マトリクス
    # vecf         : 荷重ベクトル
    # vecBoundDisp : 節点の境界条件の変位ベクトル
    # physical_field      : 全節点の変位ベクトル(np.array型)
    def set_bound_condition(self, matKt, vecf):

        matKtc = np.copy(matKt)
        vecfc = np.copy(vecf)
        vecBoundDisp = self.bound.make_disp_vector()

        # 単点拘束条件を考慮したKマトリクス、荷重ベクトルを作成する
        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                # Kマトリクスからi列を抽出する
                vecx = np.array(matKt[:, i]).flatten()

                # 変位ベクトルi列の影響を荷重ベクトルに適用する
                vecfc = vecfc - (vecBoundDisp[i]) * vecx

                # Kマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                matKtc[:, i] = 0.0
                matKtc[i, :] = 0.0
                matKtc[i, i] = 1.0
                
        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                vecfc[i] = vecBoundDisp[i]

        return matKtc, vecfc

    # 解析結果をテキストファイルに出力する
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
        f.write("\n")

        # 単点拘束情報を出力する
        f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) +"\n")
        f.write("-" * columNum * 4 + "\n")
        vecd = self.bound.make_disp_vector()
        for i in range(len(self.nodes)):
            flg = False
            for j in range(self.num_dof_at_node):
                if not vecd[self.num_dof_at_node * i + j] == None:
                    flg = True
            if flg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = str(format(vecd[self.num_dof_at_node * i], floatDigits).rjust(columNum))
                strYDisp = str(format(vecd[self.num_dof_at_node * i + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(vecd[self.num_dof_at_node * i + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")
        f.write("\n")

        # 荷重条件を出力する(等価節点力も含む)
        f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + "Z Force".rjust(columNum) +"\n")
        f.write("-" * columNum * 4 + "\n")
        vecf = self.make_force_vector()
        for i in range(len(self.nodes)):
            flg = False
            for j in range(self.num_dof_at_node):
                if not vecf[self.num_dof_at_node * i + j] == None:
                    flg = True
            if flg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[self.num_dof_at_node * i], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[self.num_dof_at_node * i + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[self.num_dof_at_node * i + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
        f.write("\n")

        # 結果データのタイトルを書きこむ
        f.write("**********************************\n")
        f.write("*          Result Data           *\n")
        f.write("**********************************\n")
        f.write("\n")

        # 変位のデータを出力する
        f.write("***** Displacement Data ******\n")
        f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Displacement".rjust(columNum) +
                "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) + "\n")
        f.write("-" * columNum * 5 + "\n")
        for i in range(len(self.nodes)):
            strNo = str(i + 1).rjust(columNum)
            mag = np.linalg.norm(np.array((self.physical_field[self.num_dof_at_node * i], self.physical_field[self.num_dof_at_node * i + 1], self.physical_field[self.num_dof_at_node * i + 2])))
            strMag = str(format(mag, floatDigits).rjust(columNum))
            strXDisp = str(format(self.physical_field[self.num_dof_at_node * i], floatDigits).rjust(columNum))
            strYDisp = str(format(self.physical_field[self.num_dof_at_node * i + 1], floatDigits).rjust(columNum))
            strZDisp = str(format(self.physical_field[self.num_dof_at_node * i + 2], floatDigits).rjust(columNum))
            f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")            
        f.write("\n")

        # 反力のデータを出力する
        f.write("***** Reaction Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + "Z Force".rjust(columNum) + "\n")
        f.write("-" * columNum * 5 + "\n")
        for i in range(len(self.nodes)):
            strNo = str(i + 1).rjust(columNum)
            mag = np.linalg.norm(np.array((self.vecRF[self.num_dof_at_node * i], self.vecRF[self.num_dof_at_node * i + 1], self.vecRF[self.num_dof_at_node * i + 2])))
            strMag = str(format(mag, floatDigits).rjust(columNum))
            strXForce = str(format(self.vecRF[self.num_dof_at_node * i], floatDigits).rjust(columNum))
            strYForce = str(format(self.vecRF[self.num_dof_at_node * i + 1], floatDigits).rjust(columNum))
            strZForce = str(format(self.vecRF[self.num_dof_at_node * i + 2], floatDigits).rjust(columNum))
            f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")            
        f.write("\n")

        # ファイルを閉じる
        f.close()
