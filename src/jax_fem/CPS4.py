import copy
import numpy as np
from functools import partial
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as JLA
from evaluate_point import EvaluatePoint
#=============================================================================
# 4節点平面応力要素のクラス
#=============================================================================
class CPS4:
    #---------------------------------------------------------------------
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 要素(Node型のリスト)
    # material        : 材料モデル
    #---------------------------------------------------------------------
    def __init__(self, id, nodes):

        # インスタンス変数を定義する
        self.id = id                               # 要素番号
        self.nodes = nodes                         # 節点の集合(Node型のリスト)
        self.material = np.empty(4, dtype=object)  # 材料モデルのリスト

        # 要素情報の設定
        self.num_node = 4                      # 節点の数
        self.num_dof_at_node = 2               # 節点の自由度

        # 積分点情報
        self.num_evaluate_points = 4

        self.evaluate_points = np.empty(8, dtype=object)
        self.evaluate_points[0] = EvaluatePoint(1, jnp.array([-jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0]))
        self.evaluate_points[1] = EvaluatePoint(2, jnp.array([ jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0]))
        self.evaluate_points[2] = EvaluatePoint(3, jnp.array([ jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0]))
        self.evaluate_points[3] = EvaluatePoint(4, jnp.array([-jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0]))

        # 要素内節点の自由度数を更新する
        for inode in range(len(self.nodes)):
            self.nodes[inode].num_dof = self.num_dof_at_node

        # 要素内の変位を初期化する
        self.Ue = jnp.zeros(self.num_node * self.num_dof_at_node)
        
        # 要素内の自由度番号のリストを作成する
        self.dof_list = self.make_dof_list(self.num_node, self.num_dof_at_node)
    
    #---------------------------------------------------------------------
    # 要素内の自由度番号のリストを作成する
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1, 2))
    def make_dof_list(self, num_node, num_dof):
        #
        dof_list = jnp.empty(num_node * num_dof, dtype=jnp.int32)
        for i in range(num_node):
            for j in range(num_dof):
                # jaxを使用する際の配列の更新方法 (dof_list[i * num_dof + j] = nodes[i].dof(j))
                dof_list = dof_list.at[i * num_dof + j].set(self.nodes[i].dof(j))
        return dof_list
    
    #---------------------------------------------------------------------
    # 材料モデルを設定
    #---------------------------------------------------------------------
    def set_material(self, material):
        for ip in range(self.num_evaluate_points):
            self.material[ip] = copy.deepcopy(material)

    #---------------------------------------------------------------------
    # 要素剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0))
    def make_Ke(self, x):
        # 初期化
        num_dof = self.num_dof_at_node * self.num_node
        Ke = jnp.zeros([num_dof, num_dof])
        # 積分点ループ
        for ip in range(self.num_evaluate_points):
            # ヤコビ行列を計算する
            J = self.make_J(ip)
            # Bマトリクスを計算する
            B = self.make_B(ip, J)
            #
            C = self.material[ip].make_C(x)
            # 重みを取得する
            weight = self.evaluate_points[ip].weight
            # 要素剛性行列Keを計算する
            Ke += weight[0] * weight[1] * B.T @ C @ B * JLA.det(J) 
        return Ke
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    def make_Fe(self):
        # 初期化
        Fe = jnp.zeros(self.num_dof_at_node * self.num_node)
        # 積分点ループ
        for ip in range(self.ipNum):
            # ヤコビ行列を計算する
            J = self.make_J(ip)
            # Bマトリクスを計算する
            B = self.make_B(ip)
            # 応力を計算する
            stress = self.material[ip].compute_stress(B, self.Ue)
            # 重みを取得する
            weight = self.num_evaluate_points[ip].weight
            # 内力ベクトルを計算する
            Fe += weight[0] * weight[1] * B.T @ stress * JLA.det(J)
        return Fe
    
    #---------------------------------------------------------------------
    # 等価節点力の荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fbe(self):
        # 初期化
        Fb = jnp.zeros(self.num_node * self.num_dof_at_node)
        # 積分点ループ
        '''for ip in range(self.ipNum):
            
            # ヤコビ行列を計算する
            matJ = self.makeJmatrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # 物体力による等価節点力を計算する
            if not self.vecGravity is None:
                vecb = self.material[ip].density * self.vecGravity   # 単位体積あたりの物体力のベクトル
                N1 = 1 - self.ai - self.bi - self.ci
                N2 = self.ai
                N3 = self.bi
                N4 = self.ci
                
                matN = np.matrix([[N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0, 0.0],
                                  [0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0],
                                  [0.0, 0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4]])
                
                Fb += self.w * np.array(matN.T @ vecb).flatten() * LA.det(matJ)'''
        return Fb
    
    #---------------------------------------------------------------------
    # ヤコビ行列を計算する
    # a : a座標値
    # b : b座標値
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1))
    def make_J(self, ip):
        # dNdabを計算する
        matdNdab = self.make_dNda(ip)
        # xi, yiの行列を計算する
        matxiyi = jnp.array([
            [self.nodes[0].coordinate[0], self.nodes[0].coordinate[1]],
            [self.nodes[1].coordinate[0], self.nodes[1].coordinate[1]],
            [self.nodes[2].coordinate[0], self.nodes[2].coordinate[1]],
            [self.nodes[3].coordinate[0], self.nodes[3].coordinate[1]]
        ])
        # ヤコビ行列を計算する
        J = matdNdab @ matxiyi
        return J

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    # a : a座標値
    # b : b座標値
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1, 2))
    def make_B(self, ip, J):
        # dNdaの行列を計算する
        dNda = self.make_dNda(ip)
        # matdNdxy = matJinv * matdNdab
        matdNdxy = JLA.solve(J, dNda)  # J @ dNdxy = dNdaを解いている
        # Bマトリクスを計算する
        B = jnp.empty((3,0))
        for i in range(self.num_node): 
            matTmp = jnp.array([
                [matdNdxy[0, i], 0.0],
                [0.0, matdNdxy[1, i]],
                [matdNdxy[1, i], matdNdxy[0, i]]
            ])
            B = jnp.hstack((B, matTmp))
        return B

    #---------------------------------------------------------------------
    # dNdxの行列を計算する
    # a : a座標値
    # b : b座標値
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1))
    def make_dNda(self, ip):
        # 積分点座標を設定する
        a = self.evaluate_points[ip].coordinaite[0]
        b = self.evaluate_points[ip].coordinaite[1]
        # dNi/da, dNi/db, dNi/dcを計算する
        dN1da = -0.25 * (1.0 - b)
        dN2da = 0.25 * (1.0 - b)
        dN3da = 0.25 * (1.0 + b)
        dN4da = -0.25 * (1.0 + b)
        dN1db = -0.25 * (1.0 - a) 
        dN2db = -0.25 * (1.0 + a) 
        dN3db = 0.25 * (1.0 + a) 
        dN4db = 0.25 * (1.0 - a) 
        # dNdaを計算する
        dNda = jnp.array([
            [dN1da, dN2da, dN3da, dN4da],
            [dN1db, dN2db, dN3db, dN4db]
        ])
        return dNda

    #---------------------------------------------------------------------
    # 構成則の計算を行う
    # elem_solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    @jit
    def compute_constitutive_law(self, elem_solution):
        
        # 要素内変位の更新
        self.solution = elem_solution

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # Bマトリックスを作成
            matB = self.make_B_matrix(ip)
            
            # 構成則の内部変数の更新
            self.material[ip].compute_stress_and_tangent_matrix(matB, elem_solution)
    
    #---------------------------------------------------------------------
    # 構成則の変数を更新する
    #---------------------------------------------------------------------
    @jit
    def update_constitutive_law(self):

        # 積分点ループ
        for ip in range(self.ipNum):
            self.material[ip].update()
