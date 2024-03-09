# kato
import os
import numpy as np

#---------------------------------------------------------------------
# 解析結果をvtkファイルに出力するクラス
# kato(20231129)
#---------------------------------------------------------------------
class Output:
    # コンストラクタ
    # nodes    : 節点は1から始まる順番で並んでいる前提(Node型のリスト)
    # elements : 要素は種類ごとにソートされている前提(CPS4型のリスト)
    # bound    : 境界条件(d2Boundary型)
    # num_step   : インクリメント数
    def __init__(self, method):
                # インスタンス変数を定義する
        self.nodes = method.nodes                  # 節点は1から始まる順番で並んでいる前提(Node2d型のリスト)
        self.elements = method.elements            # 要素は種類ごとにソートされている前提(リスト)
        self.num_step = method.num_step
        self.solution_list = method.solution_list  # インクリメント毎の変位ベクトルのリスト(np.array型のリスト)
        self.Freact_list = method.Freact_list      # インクリメント毎の反力ベクトルのリスト(np.array型のリスト)
        self.design_variable = method.design_variable  # 設計変数（np.array型のリスト）

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
            # 3Dか2Dか判定
            if self.nodes[0] == 3:
                strZ = str(format(node.z, floatDigits).rjust(columNum))
                f.write(strNo + strX + strY + strZ + "\n")
            elif self.nodes[0] == 2:
                f.write(strNo + strX + strY + "\n")
            else:
                a = 1
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
                # 3Dか2Dか判定
                if self.nodes[0] == 3:
                    strZDisp = str(format(solution[self.nodes[j].num_dof * j + 2], floatDigits).rjust(columNum))
                    f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")
                elif self.nodes[0] == 2:
                    f.write(strNo + strMag + strXDisp + strYDisp + "\n")
                else:
                    a = 1          
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
                # 3Dか2Dか判定
                if self.nodes[0] == 3:
                    strZForce = str(format(vecRF[self.nodes[j].num_dof * j + 2], floatDigits).rjust(columNum))
                    f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")
                elif self.nodes[0] == 2:
                    f.write(strNo + strMag + strXForce + strYForce + "\n")
                else:
                    a = 1
            f.write("\n")

        # ファイルを閉じる
        f.close()


    #---------------------------------------------------------------------
    # 解析結果をvtkファイルに出力する
    # kato(20231129)
    #---------------------------------------------------------------------
    def output_vtk(self, filePath):

        # ファイルを作成し、開く
        f = open(filePath + ".txt", "w")

        # 出力する文字の情報を定義する
        floatDigits = ".10g"

        # vtkファイルのバージョンを指定する
        f.write("# vtk DataFile Version 2.0\n")
        
        # データ名をつける
        f.write("FEAresults\n")
        
        # ファイルの型を指定する
        f.write("ASCII\n")
        
        # 格子タイプを非構造格子に設定する
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # 節点情報を出力する
        # POINTS 節点数 データ型
        f.write("POINTS" + " " + str(len(self.nodes)) + " " + "float\n")
        for node in self.nodes:
            strX = str(format(node.x, floatDigits))
            strY = str(format(node.y, floatDigits))
            strZ = str(format(0.0, floatDigits))
            f.write(strX + " " + strY + " " + strZ + "\n")
            
        # 要素のコネクティビティの情報を出力する
        # CELLS 要素数 データの数
        num_nodes = self.elements[0].num_node
        f.write("CELLS" + " " + str(len(self.elements)) + " " + str(len(self.elements) * (num_nodes + 1)) + "\n")
        # 要素の節点数 節点ID1 節点ID2 節点ID3 節点ID4
        for element in self.elements:
            strNumNodes = str(num_nodes)
            strNodeID1 = str(element.nodes[0].no - 1)
            strNodeID2 = str(element.nodes[1].no - 1)
            strNodeID3 = str(element.nodes[2].no - 1)
            strNodeID4 = str(element.nodes[3].no - 1)
            f.write(strNumNodes + " " + strNodeID1 + " " + strNodeID2 + " " + strNodeID3 + " " + strNodeID4 + "\n")
        
        # 要素タイプの設定
        f.write("CELL_TYPES" + " " + str(len(self.elements)) + "\n")
        # 四角形要素は9の番号が割り当てられている
        # 他の要素は公式のpdfを参照 "https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf"
        for element in self.elements:
            f.write(str(9) + "\n")
        
        # 結果データを作成する
        for i in range(self.num_step):
            # 変位のデータを出力する
            # POINT_DATA 節点数
            f.write("POINT_DATA" + " " + str(len(self.nodes)) + "\n")
            f.write("VECTORS DISP float\n")
            for j in range(len(self.nodes)):
                solution = self.solution_list[i]
                strXDisp = str(format(solution[self.nodes[j].num_dof * j], floatDigits))
                strYDisp = str(format(solution[self.nodes[j].num_dof * j + 1], floatDigits))
                strZDisp = str(format(0.0, floatDigits))
                f.write(strXDisp + " " + strYDisp + " " + strZDisp + "\n")
        
            # 反力のデータを出力する
            f.write("VECTORS Freact float\n")
            for j in range(len(self.nodes)):
                vecRF = self.Freact_list[i]
                strXforce = str(format(vecRF[self.nodes[j].num_dof * j], floatDigits))
                strYforce = str(format(vecRF[self.nodes[j].num_dof * j + 1], floatDigits))
                strZforce = str(format(0.0, floatDigits))
                f.write(strXforce + " " + strYforce + " " + strZforce + "\n")
                
            # 設計変数のデータを出力する
            f.write("CELL_DATA" + " " + str(len(self.elements)) + "\n")
            f.write("SCALARS Design_Variable float\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(len(self.elements)):
                strDesignVariable = str(format(self.design_variable[j], floatDigits))
                f.write(strDesignVariable + "\n")
                
            f.write("SCALARS Young float\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(len(self.elements)):
                total_young = 0
                for k in range(self.elements[j].ipNum):
                    total_young = total_young + self.elements[j].material[k].young
                
                average_young = total_young / self.elements[j].ipNum
                
                strAveYoung = str(format(average_young, floatDigits))
                f.write(strAveYoung + "\n")
        
        #ファイルを閉じる
        f.close()

        # txtファイルをvtkファイルに変更
        os.rename(filePath + ".txt", filePath + ".vtk")
