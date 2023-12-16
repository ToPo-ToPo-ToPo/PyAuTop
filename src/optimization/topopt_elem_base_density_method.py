from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import copy
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
root_dir = dirname(parent_dir)
from analysis.analysis import Analysis
from output.output import Output
import time

def main(volfrac, penal, rfil, ft):
    
    # Analysis classの作成のための変数
    ID = "1"
    ANALYSIS_TYPE = "Main"
    MESH_TYPE = "Auto"
    PHYSICS_TYPE = "Static_Structure"
    METHOD_TYPE = "Nonlinear_FEM"
    NUM_STEP = "1"
    
    CONV = 0.01 # 収束判定のためのパラメータ
    MAX_STEP = 20 # トポロジー最適化のループを回す最大回数
    
    itr = 0
    min_compliance = 1e20  # コンプライアンス（目的関数）
    convflag=0
    
    # 解析クラスの作成
    analysis = Analysis(ID, ANALYSIS_TYPE, MESH_TYPE, PHYSICS_TYPE, METHOD_TYPE, NUM_STEP)
    # 解析クラスの中の物理モデル
    model = analysis.physics
    
    nx = model.xdiv
    ny = model.ydiv
    
    # 解析条件の表示
    print("Minimum compliance problem with OC")
    print("Design domain shape: " + str(nx) + "x" + str(ny))
    print("Volume constrain: " + str(volfrac) + ", Filter radius" + str(rfil) + ", SIMP penal: " + str(penal))
    print("Filter method: " +  ["Sensitivity based", "Density based"][ft])
    
    # 設計変数の定義
    design_variable = volfrac * np.ones(ny*nx, dtype=float)
    design_variable_old = copy.deepcopy(design_variable)
    
    # 目的関数の感度と体積の感度を用意しておく
    compliance_sensitivity = np.ones(ny*nx)
    volume_sensitivity = np.ones(ny*nx)
            
    # ループ開始時刻
    start = time.time()
    
    for optstep in range(MAX_STEP):
        
        # ステータスの表示
        print("======================== Optimization step =========================")
        print(f" STEP NO.{optstep}")

        #---------------------------------------------------------------------
        # 材料分布からヤング率を計算する
        #---------------------------------------------------------------------
        for i in range (len(model.elems)):
            # 積分点のループ
            for ip in range(model.elems[i].ipNum):
                model.elems[i].material[ip].update_parameter(design_variable[i])
        
        #---------------------------------------------------------------------
        # FEMの初期化 -> NonlinearFEMをrunするときにsolution_listに空の配列を代入しているので必要ないかも（定常じゃない問題は対応できないけど今のところOK）
        # FEMの実行
        #---------------------------------------------------------------------
        analysis.method.run()
        analysis.method.design_variable = design_variable
        
        #---------------------------------------------------------------------
        # 最適化結果の出力(vkt)もここで行われる
        #---------------------------------------------------------------------
        Output(analysis.method).output_vtk(root_dir +  "/output/CPS4_test")
        
        #---------------------------------------------------------------------
        # 目的関数と制約関数の評価
        #---------------------------------------------------------------------
        disp = analysis.method.solution_list[0] # 変位
        compliance = disp.T @ analysis.method.Kt @ disp # コンプライアンス
        v = np.mean(design_variable) # 正確には数値積分が必要

        #---------------------------------------------------------------------
        # 感度解析
        # 感度フィルター
        #---------------------------------------------------------------------   
        # コンプライアンスの感度解析
        for i in range (len(model.elems)):
            dKe = model.elems[i].make_dK(design_variable[i], penal)
            compliance_sensitivity[i] = - model.elems[i].solution.T @ dKe @ model.elems[i].solution
        
        # 感度フィルター
        compliance_sensitivity = sensitivity_filter(nx, ny, rfil, design_variable, compliance_sensitivity)
        volume_sensitivity = np.ones(ny*nx) # 体積の感度
        
        #---------------------------------------------------------------------
        # 設計変数の更新（OC法）
        #--------------------------------------------------------------------- 
        design_variable = OC(volfrac, design_variable, compliance_sensitivity)
        
        #---------------------------------------------------------------------
        # 収束判定
        #---------------------------------------------------------------------        
        # 常にコンプライアンスが小さくなるようにしている
        if min_compliance > compliance:
            min_compliance = compliance
            convflag = 0
        else:
            convflag = convflag+1
        
        # 設計変数の最大の変化量が収束パラメータより小さい時にループを終わらせる
        change = np.max(np.abs(design_variable - design_variable_old))
        if change < CONV:
            break
        
        # 次の収束判定のために一つ前の設計変数を保存しておく
        design_variable_old = design_variable
    
    # ループ終了時刻
    end = time.time()
    print(str((end-start)/60) + " minutes")

#---------------------------------------------------------------------
# 感度フィルター
#---------------------------------------------------------------------    
def sensitivity_filter(nx, ny, rfil, design_variable, compliance_sensitivity):
    
    filtered_c_sens = np.zeros((ny, nx)) # フィルターをかけた目的関数の感度
    
    design_variable = design_variable.reshape((ny, nx))
    compliance_sensitivity = compliance_sensitivity.reshape(ny,nx)
    
    # 設計領域に関するループ
    for iy in range(ny):
        for ix in range(nx):
            sum = 0.0
            
            # フィルターのループ（畳み込み演算）
            for i in range(max(int(ix-np.ceil(rfil)), 0), min(int(ix+np.ceil(rfil)), nx)):
                for j in range(max(int(iy-np.ceil(rfil)), 0), min(int(iy+np.ceil(rfil)), ny)):
                    tmp = rfil - np.sqrt((ix-i)**2 + (iy-j)**2)
                    sum = sum + max(0, tmp)
                    filtered_c_sens[iy, ix] = filtered_c_sens[iy, ix] + max(0, tmp) * design_variable[j,i] * compliance_sensitivity[j,i]
            
            filtered_c_sens[iy, ix] = filtered_c_sens[iy, ix]/max(design_variable[iy,ix]*sum, 1e-10)
    
    filtered_c_sens = filtered_c_sens.reshape(ny*nx)
            
    return filtered_c_sens
    

#---------------------------------------------------------------------
# OC法
#---------------------------------------------------------------------    
def OC(volfrac, design_variable, compliance_sensitivity):
    lambda1 = 0
    lambda2 = 1e4
    mvlmt = 0.15 # ムーブリミット（設計変数の変動上限）
    eta = 0.5 # ダンピング係数
    # ラグランジュ定数が体積制約を満たすように二分法を用いて探索する
    while (lambda2-lambda1)/(lambda1+lambda2) > 1e-3:
        lambda_mid = (lambda2 + lambda1) * 0.5
        tmp = design_variable * (-1 * compliance_sensitivity / lambda_mid) ** eta
        # 以下の操作では必ず設計変数は0~1の値をとる
        new_design_variable =np.maximum(np.maximum(np.minimum(np.minimum(tmp, design_variable + mvlmt), 1), design_variable - mvlmt), 0)
        if np.mean(new_design_variable) - volfrac > 0:
            lambda1 = lambda_mid
        else:
            lambda2 = lambda_mid
    return new_design_variable

# The real main driver    
if __name__ == "__main__":
	# Default input parameters
	# nelx=180
	# nely=60
	volfrac=0.4
	rfil=1
	penal=3.0
	ft=1 # ft==0 -> sens, ft==1 -> dens
	import sys
	# if len(sys.argv)>1: nelx   =int(sys.argv[1])
	# if len(sys.argv)>2: nely   =int(sys.argv[2])
	if len(sys.argv)>3: volfrac=float(sys.argv[3])
	if len(sys.argv)>4: rfil   =float(sys.argv[4])
	if len(sys.argv)>5: penal  =float(sys.argv[5])
	if len(sys.argv)>6: ft     =int(sys.argv[6])
	main(volfrac,penal,rfil,ft)