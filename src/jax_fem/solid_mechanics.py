
from functools import partial
import jax.numpy as jnp
from jax import jit
from boundary_condition import BoundaryConditions, NeummanBc
from C3D8 import C3D8
# =============================================================================
# 固体力学のクラス
# =============================================================================
class SolidMechanics:
    #---------------------------------------------------------------------
    # コンストラクタ
    #---------------------------------------------------------------------
    def __init__(self, nodes, elements, boundary_conditions):
        # インスタンス変数を定義する
        self.nodes = nodes
        self.elements = elements
        self.boundary_conditions = boundary_conditions
        
        # 方程式の数を計算
        self.num_total_equation = self.compute_num_total_equation()
    
    #---------------------------------------------------------------------
    # 総自由度数を計算する
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0))
    def compute_num_total_equation(self):
        num_total_equation = 0
        for i in range(len(self.nodes)):
            num_total_equation += self.nodes[i].num_dof
        return num_total_equation
    
    #---------------------------------------------------------------------
    # 接線剛性マトリクスKtを作成する
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0))
    def make_K(self):
        # 初期化
        K = jnp.array(jnp.zeros((self.num_total_equation, self.num_total_equation)))
        # 全要素ループ
        for elem in self.elements:
            # ketマトリクスを計算する
            Ke = elem.make_Ke()
            # Ktマトリクスに代入する
            for c in range(len(elem.dof_list)):
                # 自由度番号の取得
                ct = elem.dof_list[c]
                for r in range(len(elem.dof_list)):
                    # 自由度番号の取得
                    rt = elem.dof_list[r]
                    # アセンブリング: jaxを使用する際の配列の更新方法
                    K = K.at[(ct, rt)].add(Ke[c, r])
        return K
    
    #---------------------------------------------------------------------
    # 表面力による外力ベクトルを作成
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1))
    def make_Ft(self, istep):
        # 初期化
        F = jnp.zeros(self.num_total_equation)
        for bc in self.boundary_conditions.neumman_bcs:
            values = bc.values[istep, :]
            for node in bc.nodes:
                for i in range(node.num_dof):
                    F = F.at[node.dof(i)].add(values[i])                     
        return F
    
    #---------------------------------------------------------------------
    # 物体力による外力ベクトルを作成
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1))
    def make_Fb(self, istep):
        # 初期化
        F = jnp.zeros(self.num_total_equation)    
        # 全要素ループ
        for elem in self.elements:
            # 要素物体力ベクトルを作成する
            Fe = elem.make_Fb()   
            # アセンブリング
            for i in range(len(elem.dof_list)):
                F = F.at[elem.dof_list[i]].add(Fe[i])
        return F
    
    #---------------------------------------------------------------------
    # ディレクレ境界用にデータを方程式を更新
    #---------------------------------------------------------------------
    #@partial(jit, static_argnums=(0, 1, 2, 3, 4))
    def consider_dirichlet_bc(self, istep, lhs, rhs, U):

        # 初期化
        lhs_c = jnp.copy(lhs)
        rhs_c = jnp.copy(rhs)
        
        # 拘束条件をのループ
        for bc in self.boundary_conditions.dirichlet_bcs:
            # データの取得
            flags = bc.flags
            values = bc.values[istep, :]
            # 節点のループ
            for node in bc.nodes:
                for i in range(node.num_dof):
                    # check
                    flag_equals_1 = jnp.equal(flags[i], 1)
                    if flag_equals_1:
                    #if flags[i] == 1:
                        # 条件を与える場合
                        dof = node.dof(i)
                        # Kマトリクスからi列を抽出する
                        vecx = jnp.array(lhs_c[:, dof]).flatten()
                        # 変位ベクトルi列の影響を荷重ベクトルに適用する
                        rhs_c = rhs_c - (values[i] - U[dof]) * vecx
                        # Kマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                        lhs_c = lhs_c.at[:,   dof].set(0.0)
                        lhs_c = lhs_c.at[dof,   :].set(0.0)
                        lhs_c = lhs_c.at[dof, dof].set(1.0)
                        # 強制値を右辺へ設定
                        rhs_c = rhs_c.at[dof].set(values[i] - U[dof])
        return lhs_c, rhs_c
    
    