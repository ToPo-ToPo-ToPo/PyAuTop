
import abc
import numpy as np
import numpy.linalg as LA

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
        K = np.matrix(np.zeros((len(self.nodes) * self.num_dof_at_node, len(self.nodes) * self.num_dof_at_node)))
        
        # 全要素ループ
        for elem in self.elements:

            # ketマトリクスを計算する
            Ke = elem.makeKematrix()

            # Ktマトリクスに代入する
            for c in range(len(elem.nodes) * self.num_dof_at_node):
                
                # 自由度番号の取得
                ct = (elem.nodes[c // self.num_dof_at_node].no - 1) * self.num_dof_at_node + c % self.num_dof_at_node
                
                for r in range(len(elem.nodes) * self.num_dof_at_node):
                    
                    # 自由度番号の取得
                    rt = (elem.nodes[r // self.num_dof_at_node].no - 1) * self.num_dof_at_node + r % self.num_dof_at_node
                    
                    # アセンブリング
                    K[ct, rt] += Ke[c, r]

        return K

    #---------------------------------------------------------------------
    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fext(self):

        # 節点に負荷する荷重ベクトルを作成する
        Ft = self.bound.make_Ft()

        # 等価節点力の荷重ベクトルを作成する
        Fb = np.zeros(len(self.nodes) * self.num_dof_at_node)
        
        # 全要素ループ
        for elem in self.elements:
            
            # 要素物体力ベクトルを作成する
            Fb_e = elem.make_Fb()
            
            # アセンブリング
            for i in range(len(elem.nodes)):
                for j in range(self.num_dof_at_node):
                    Fb[self.num_dof_at_node * (elem.nodes[i].no - 1) + j] += Fb_e[self.num_dof_at_node * i + j]

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
