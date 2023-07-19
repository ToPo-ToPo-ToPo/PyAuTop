#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np
import numpy.linalg as LA

class FEM:
    # コンストラクタ
    # nodes    : 節点は1から始まる順番で並んでいる前提(Node型のリスト)
    # elements : 要素は種類ごとにソートされている前提(C3D８型のリスト)
    # bound    : 境界条件(d2Boundary型)
    # incNum   : インクリメント数
    def __init__(self, nodes, elements, bound, incNum):

        # インスタンス変数を定義する
        self.nodeDof = 3           # 節点の自由度
        self.nodes = nodes         # 節点は1から始まる順番で並んでいる前提(Node2d型のリスト)
        self.elements = elements   # 要素は種類ごとにソートされている前提(リスト)
        self.bound = bound         # 境界条件(d2Boundary型)
        self.incNum = incNum       # インクリメント数
        self.itrNum = 100          # ニュートン法のイテレータの上限
        self.rn = 0.005            # ニュートン法の残差力の収束判定のパラメータ
        self.cn = 0.01             # ニュートン法の変位の収束判定のパラメータ

    # 陰解法で解析を行う
    def impAnalysis(self):

        self.vecDispList = []          # インクリメント毎の変位ベクトルのリスト(np.array型のリスト)
        self.vecRFList = []            # インクリメント毎の反力ベクトルのリスト(np.array型のリスト)
        self.elemOutputDataList = []   # インクリメント毎の要素出力のリスト(makeOutputData型のリストのリスト)

        # 荷重をインクリメント毎に分割する
        vecfList = []
        for i in range(self.incNum):
            vecfList.append(self.makeForceVector() * (i + 1) / self.incNum)

        # ニュートン法により変位を求める
        vecDisp = np.zeros(len(self.nodes) * self.nodeDof)   # 全節点の変位ベクトル
        vecR = np.zeros(len(self.nodes) * self.nodeDof)      # 残差力ベクトル
        for i in range(self.incNum):
            vecDispFirst = vecDisp.copy()   # 初期の全節点の変位ベクトル
            vecf = vecfList[i]              # i+1番インクリメントの荷重
            vecBoundDisp = self.bound.makeDispVector()

            # 境界条件を考慮しないインクリメント初期の残差力ベクトルRを作成する
            if i == 0:
                vecR = vecfList[0]
            else:
                vecR = vecfList[i] - vecfList[i - 1]

            # 収束演算を行う
            for j in range(self.itrNum):

                # 接線剛性マトリクスKtを作成する
                matKt = self.makeKtmatrix()

                # 境界条件を考慮したKtcマトリクス、Rcベクトルを作成する
                matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)

                # Ktcの逆行列が計算できるかチェックする
                if np.isclose(LA.det(matKtc), 0.0) :
                    raise ValueError("有限要素法の計算に失敗しました。")

                # 変位ベクトルを計算する
                vecd = LA.solve(matKtc, vecRc)
                vecDisp += vecd

                # 全ての要素内の変数を更新する
                self.updateElements(vecDisp, i)

                # 新たな残差力ベクトルRを求める
                vecQ = np.zeros(len(self.nodes) * self.nodeDof)
                for elem in self.elements:
                    vecq = elem.makeqVector()
                    for k in range(len(elem.nodes)):
                        for l in range(elem.nodeDof):
                            vecQ[(elem.nodes[k].no - 1) * self.nodeDof + l] += vecq[k * elem.nodeDof + l]
                vecR = vecf - vecQ

                # 新たな境界条件を考慮したRcベクトルを作成する
                matKt = self.makeKtmatrix()
                matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)

                # 時間平均力を計算する
                aveForce = 0.0
                cnt = len(self.nodes) * self.nodeDof
                for k in range(len(vecQ)):
                    aveForce += np.abs(vecQ[k])
                for k in range(len(vecf)):
                    if not vecf[k] == 0.0:
                        aveForce += np.abs(vecf[k])
                        cnt+= 1
                aveForce = aveForce / cnt

                # 収束判定を行う
                if np.allclose(vecRc, 0.0):
                    break
                if np.isclose(LA.norm(vecd), 0.0):
                    break
                dispRate = LA.norm(vecd) / LA.norm(vecDisp - vecDispFirst)
                ResiForceRate = np.abs(vecRc).max() / aveForce
                if dispRate < self.cn and ResiForceRate < self.rn:
                    break

            # インクリメントの最終的な変位べクトルを格納する
            self.vecDispList.append(vecDisp.copy())

            # 変位ベクトルから要素の出力データを計算する
            elemOutputDatas = []
            for elem in self.elements:
                elemOutputData = elem.makeOutputData()
                elemOutputDatas.append(elemOutputData)        
            self.elemOutputDataList.append(elemOutputDatas)

            # 節点反力を計算する
            vecRF = np.array(vecQ - vecf).flatten()

            # インクリメントの最終的な節点反力を格納する
            self.vecRFList.append(vecRF)      

    # 接線剛性マトリクスKtを作成する
    def makeKtmatrix(self):

        matKt = np.matrix(np.zeros((len(self.nodes) * self.nodeDof, len(self.nodes) * self.nodeDof)))
        for elem in self.elements:

            # ketマトリクスを計算する
            matKet = elem.makeKetmatrix()

            # Ktマトリクスに代入する
            for c in range(len(elem.nodes) * self.nodeDof):
                ct = (elem.nodes[c // self.nodeDof].no - 1) * self.nodeDof + c % self.nodeDof
                for r in range(len(elem.nodes) * self.nodeDof):
                    rt = (elem.nodes[r // self.nodeDof].no - 1) * self.nodeDof + r % self.nodeDof
                    matKt[ct, rt] += matKet[c, r]

        return matKt

    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    def makeForceVector(self):

        # 節点に負荷する荷重ベクトルを作成する
        vecCondiForce = self.bound.makeForceVector()
        vecf = vecCondiForce

        return vecf


    # Kマトリクス、荷重ベクトルに境界条件を考慮する
    # matKt        : 接線剛性マトリクス
    # vecR         : 残差力ベクトル
    # vecBoundDisp : 節点の境界条件の変位ベクトル
    # vecDisp      : 全節点の変位ベクトル(np.array型)
    def setBoundCondition(self, matKt, vecR, vecBoundDisp, vecDisp):

        matKtc = np.copy(matKt)
        vecRc = np.copy(vecR)

        # 単点拘束条件を考慮したKマトリクス、荷重ベクトルを作成する
        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                # Kマトリクスからi列を抽出する
                vecx = np.array(matKt[:, i]).flatten()

                # 変位ベクトルi列の影響を荷重ベクトルに適用する
                vecRc = vecRc - (vecBoundDisp[i] - vecDisp[i]) * vecx

                # Kマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                matKtc[:, i] = 0.0
                matKtc[i, :] = 0.0
                matKtc[i, i] = 1.0

        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                vecRc[i] = vecBoundDisp[i] - vecDisp[i]

        return matKtc, vecRc

    # 全ての要素内の変数を更新する
    # vecDisp : 全節点の変位ベクトル(np.array型)
    # incNo   : インクリメントの番号
    def updateElements(self, vecDisp, incNo):

        for elem in self.elements:
            vecElemDisp = np.zeros(len(elem.nodes) * self.nodeDof)
            for i in range(len(elem.nodes)):
                for j in range(elem.nodeDof):
                    vecElemDisp[i * elem.nodeDof + j] = vecDisp[(elem.nodes[i].no - 1) * self.nodeDof + j]
            elem.update(vecElemDisp, incNo) 

    # 解析結果をテキストファイルに出力する
    def outputTxt(self, filePath):

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
                    strThickness + strArea + strDensity + "\n")
        f.write("\n")

        # 単点拘束情報を出力する
        f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "Y Displacement".rjust(columNum) + 
                "Z Displacement".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecBoundDisp = self.bound.makeDispVector()
        for i in range(self.bound.nodeNum):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecBoundDisp[i * self.bound.nodeDof + j] is None:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof] is None:
                    strXDisp = str(format(vecBoundDisp[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 1] is None:
                    strYDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 2] is None:
                    strZDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")
        f.write("\n")

        # 荷重条件を出力する(等価節点力も含む)
        f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                "Z Force".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecf = self.makeForceVector()
        for i in range(len(self.nodes)):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecf[i * self.bound.nodeDof + j] == 0.0:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
        f.write("\n")

        # 結果データのタイトルを書きこむ
        f.write("**********************************\n")
        f.write("*          Result Data           *\n")
        f.write("**********************************\n")
        f.write("\n")

        for i in range(self.incNum):
            f.write("*Increment " + str(i + 1) + "\n")
            f.write("\n")

            # 変位のデータを出力する
            f.write("***** Displacement Data ******\n")
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Displacement".rjust(columNum) +
                    "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecDisp = self.vecDispList[i]
                mag = np.linalg.norm(np.array((vecDisp[self.nodeDof * j], vecDisp[self.nodeDof * j + 1], vecDisp[self.nodeDof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXDisp = str(format(vecDisp[self.nodeDof * j], floatDigits).rjust(columNum))
                strYDisp = str(format(vecDisp[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(vecDisp[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")            
            f.write("\n")

            # 応力データを出力する
            f.write("***** Stress Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "Stress XX".rjust(columNum) + "Stress YY".rjust(columNum) + 
                    "Stress ZZ".rjust(columNum) + "Stress XY".rjust(columNum) + "Stress XZ".rjust(columNum) + "Stress YZ".rjust(columNum) +
                    "Mises".rjust(columNum) + "\n")
            f.write("-" * columNum * 9 + "\n")
            for elemOutputData in self.elemOutputDataList[i]:
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
            for elemOutputData in self.elemOutputDataList[i]:
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
            for elemOutputData in self.elemOutputDataList[i]:
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
            f.write("\n")           

            # 反力のデータを出力する
            f.write("***** Reaction Force Data ******\n")
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                    "Z Force".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecRF = self.vecRFList[i]
                mag = np.linalg.norm(np.array((vecRF[self.nodeDof * j], vecRF[self.nodeDof * j + 1], vecRF[self.nodeDof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXForce = str(format(vecRF[self.nodeDof * j], floatDigits).rjust(columNum))
                strYForce = str(format(vecRF[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecRF[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")            
            f.write("\n")

        # ファイルを閉じる
        f.close()
