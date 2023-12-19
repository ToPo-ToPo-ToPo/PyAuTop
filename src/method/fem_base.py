
import abc
import numpy as np
import numpy.linalg as LA
from scipy.sparse import lil_matrix
from concurrent import futures

#=============================================================================
# 有限要素法の基本クラス
# インターフェースの機能を持ち、具体的な実装は継承先にて行う
#=============================================================================
class FEMInterface(metaclass=abc.ABCMeta):

    #---------------------------------------------------------------------
    # 解析を実行する
    #---------------------------------------------------------------------
    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError()

#=============================================================================
# 有限要素法の基本クラス
#=============================================================================
class FEMBase(FEMInterface):
    
    
    #---------------------------------------------------------------------
    # 解析を実行する
    #---------------------------------------------------------------------
    def run(self) -> None:
        pass

    #---------------------------------------------------------------------
    # 接線剛性マトリクスKtを作成する
    #---------------------------------------------------------------------
    def make_K(self):

        # 初期化
        K = np.matrix(np.zeros((self.num_total_equation, self.num_total_equation)))
        
        # 全要素ループ
        for elem in self.elements:

            # ketマトリクスを計算する
            Ke = elem.make_K()

            # Ktマトリクスに代入する
            for c in range(len(elem.dof_list)):
                
                # 自由度番号の取得
                ct = elem.dof_list[c]
                
                for r in range(len(elem.dof_list)):
                    
                    # 自由度番号の取得
                    rt = elem.dof_list[r]
                    
                    # アセンブリング
                    K[ct, rt] += Ke[c, r]

        return K
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    def make_Fint(self):

        # 初期化
        Fint = np.zeros(self.num_total_equation)
        
        # 全要素ループ
        for elem in self.elements:
            
            # 要素内力ベクトルを作成する
            Fe = elem.make_Fint()
            
            # アセンブリング
            for i in range(len(elem.dof_list)):
                Fint[elem.dof_list[i]] += Fe[i]
        
        return Fint
    
    #---------------------------------------------------------------------
    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fext(self):

        # 節点に負荷する荷重ベクトルを作成する
        Ft = self.bound.make_Ft()

        # 等価節点力の荷重ベクトルを作成する
        Fb = np.zeros(self.num_total_equation)
        
        # 全要素ループ
        for elem in self.elements:
            
            # 要素物体力ベクトルを作成する
            Fb_e = elem.make_Fb()
            
            # アセンブリング
            for i in range(len(elem.dof_list)):
                Fb[elem.dof_list[i]] += Fb[i]

        # 境界条件、等価節点力の荷重ベクトルを足し合わせる
        Fext = Ft + Fb

        return Fext
    
    #---------------------------------------------------------------------
    # Kマトリクス、荷重ベクトルに境界条件を考慮する
    # Kt            : 接線剛性マトリクス
    # R             : 残差力ベクトル
    # solution_bar  : 節点の境界条件値のベクトル
    # solution      : 全節点の解ベクトル(np.array型)
    #---------------------------------------------------------------------
    def set_bound_condition(self, lhs, rhs, solution_bar, solution):

        # 初期化
        lhs_c = np.copy(lhs)
        rhs_c = np.copy(rhs)

        # 単点拘束条件を考慮したKマトリクス、荷重ベクトルを作成する
        for i in range(len(solution_bar)):
            if not solution_bar[i] == None:
                
                # Kマトリクスからi列を抽出する
                vecx = np.array(lhs_c[:, i]).flatten()

                # 変位ベクトルi列の影響を荷重ベクトルに適用する
                rhs_c = rhs_c - (solution_bar[i] - solution[i]) * vecx

                # Kマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                lhs_c[:, i] = 0.0
                lhs_c[i, :] = 0.0
                lhs_c[i, i] = 1.0

        for i in range(len(solution_bar)):
            if not solution_bar[i] == None:
                rhs_c[i] = solution_bar[i] - solution[i]

        return lhs_c, rhs_c
    
    #---------------------------------------------------------------------
    # 全ての要素内の変数を更新する
    # solution : 全節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def update_element_data(self, solution):

        # 全要素ループ
        for elem in self.elements:
            
            # 要素状態場の初期化
            elem_solution = np.zeros(elem.num_node * elem.num_dof_at_node)
            
            # 要素の状態場を更新する
            for i in range(len(elem.dof_list)):
                elem_solution[i] = solution[elem.dof_list[i]]

            # 構成則の内部の変数を更新する
            elem.compute_constitutive_law(elem_solution) 
    
    #---------------------------------------------------------------------
    # 全ての要素内の変数を更新する
    #---------------------------------------------------------------------
    def update_constitutive_low(self):

        # 全要素ループ
        for elem in self.elements:

            # 構成則の内部の変数を更新する
            elem.update_constitutive_law() 

    #---------------------------------------------------------------------
    # 総自由度数を計算する
    #---------------------------------------------------------------------
    def compute_num_total_equation(self):
        
        self.num_total_equation = 0
        
        for i in range(len(self.nodes)):
            self.num_total_equation += self.nodes[i].num_dof