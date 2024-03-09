import copy
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as JLA
from evaluate_point import EvaluatePoint
#=============================================================================
# 6面体8節点要素のクラス
#=============================================================================
class C3D8:
    #---------------------------------------------------------------------
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 要素(Node型のリスト)
    # material        : 材料モデル
    #---------------------------------------------------------------------
    def __init__(self, id, nodes, material):

        # インスタンス変数を定義する
        self.id = id                           # 要素番号
        self.nodes = nodes                     # 節点の集合(Node型のリスト)
        self.material = []                     # 材料モデルのリスト
        
        # 要素情報の設定
        self.num_node = 8                      # 節点の数
        self.num_dof_at_node = 3               # 節点の自由度

        # 積分点情報
        self.num_evaluate_points = 8
        
        # 積分点の情報を追加していく
        self.evaluate_points = []
        self.evaluate_points.append(EvaluatePoint(1, jnp.array([-jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(2, jnp.array([ jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(3, jnp.array([ jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(4, jnp.array([-jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(5, jnp.array([-jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(6, jnp.array([ jnp.sqrt(1.0 / 3.0), -jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(7, jnp.array([ jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        self.evaluate_points.append(EvaluatePoint(8, jnp.array([-jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0),  jnp.sqrt(1.0 / 3.0)]), jnp.array([1.0, 1.0, 1.0])))
        
        # 要素内節点の自由度数を更新する
        for inode in range(len(self.nodes)):
            self.nodes[inode].num_dof = self.num_dof_at_node

        # 要素内の変位を初期化する
        self.Ue = jnp.zeros(self.num_node * self.num_dof_at_node)
        
        # 要素内の自由度番号のリストを作成する
        self.dof_list = self.make_dof_list(nodes, self.num_node, self.num_dof_at_node)

        # 材料モデルを初期化する
        for ip in range(self.ipNum):
            self.material.append(copy.deepcopy(material))

    #---------------------------------------------------------------------
    # 要素接線剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    @jit
    def make_Ke(self):
        # 初期化
        num_dof = self.num_dof_at_node * self.num_node
        Ke = jnp.zeros([num_dof, num_dof])
        # 積分点ループ
        for ip in range(self.ipNum):
            # ヤコビ行列を計算する
            J = self.make_J(ip)
            # Bマトリクスを計算する
            B = self.make_B(ip)
            #
            C = self.material[ip].make_C()
            # 重みを取得する
            weight = self.num_evaluate_points[ip].weight
            # 要素剛性行列Keを計算する
            Ke += weight[0] * weight[1] * weight[2] * B.T @ C @ B * JLA.det(J)  
        return Ke
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    def make_Fint(self):
        # 初期化
        Fint_e = jnp.zeros(self.num_dof_at_node * self.num_node)
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
            Fint_e += weight[0] * weight[1] * weight[2] * B.T @ stress * JLA.det(J)
        return Fint_e
    
    #---------------------------------------------------------------------
    # 等価節点力の荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fb(self):
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
    # c : c座標値
    #---------------------------------------------------------------------
    @jit
    def make_J(self, ip):
        # dNdabを計算する
        matdNdabc = self.make_dNda(ip)
        # xi, yi, ziの行列を計算する
        matxiyizi = jnp.array([
            [self.nodes[0].coordinaite[0], self.nodes[0].coordinaite[1], self.nodes[0].coordinaite[2]],
            [self.nodes[1].coordinaite[0], self.nodes[1].coordinaite[1], self.nodes[1].coordinaite[2]],
            [self.nodes[2].coordinaite[0], self.nodes[2].coordinaite[1], self.nodes[2].coordinaite[2]],
            [self.nodes[3].coordinaite[0], self.nodes[3].coordinaite[1], self.nodes[3].coordinaite[2]],
            [self.nodes[4].coordinaite[0], self.nodes[4].coordinaite[1], self.nodes[4].coordinaite[2]],
            [self.nodes[5].coordinaite[0], self.nodes[5].coordinaite[1], self.nodes[5].coordinaite[2]],
            [self.nodes[6].coordinaite[0], self.nodes[6].coordinaite[1], self.nodes[6].coordinaite[2]],
            [self.nodes[7].coordinaite[0], self.nodes[7].coordinaite[1], self.nodes[7].coordinaite[2]]
        ])
        # ヤコビ行列を計算する
        J = matdNdabc @ matxiyizi
        # ヤコビアンが負にならないかチェックする
        if JLA.det(J) < 0.0:
            raise ValueError("要素の計算に失敗しました")
        return J

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    @jit
    def make_B(self, ip):
        # dNdaの行列を計算する
        dNda = self.make_dNda(ip)
        # ヤコビ行列を計算する
        J = self.make_J(ip)
        # matdNdxyz = matJinv * matdNdabc
        matdNdxyz = JLA.solve(J, dNda)

        # Bマトリクスを計算する
        B = jnp.empty((6,0))
        for i in range(self.num_node): 
            matTmp = jnp.array([
                [matdNdxyz[0, i], 0.0, 0.0],
                [0.0, matdNdxyz[1, i], 0.0],
                [0.0, 0.0, matdNdxyz[2, i]],
                [0.0, matdNdxyz[2, i], matdNdxyz[1, i]],
                [matdNdxyz[2, i], 0.0, matdNdxyz[0, i]], 
                [matdNdxyz[1, i], matdNdxyz[0, i], 0.0]
            ]) 
            B = jnp.hstack((B, matTmp))

        return B

    #---------------------------------------------------------------------
    # dNdxの行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    @jit
    def make_dNda(self, ip):
        # 積分点座標を設定する
        a = self.evaluate_points[ip].coordinaite[0]
        b = self.evaluate_points[ip].coordinaite[1]
        c = self.evaluate_points[ip].coordinaite[2]
        # dNi/da, dNi/db, dNi/dcを計算する
        dN1da = -0.125 * (1.0 - b) * (1.0 - c)
        dN2da = 0.125 * (1.0 - b) * (1.0 - c)
        dN3da = 0.125 * (1.0 + b) * (1.0 - c)
        dN4da = -0.125 * (1.0 + b) * (1.0 - c)
        dN5da = -0.125 * (1.0 - b) * (1.0 + c)
        dN6da = 0.125 * (1.0 - b) * (1.0 + c)
        dN7da = 0.125 * (1.0 + b) * (1.0 + c)
        dN8da = -0.125 * (1.0 + b) * (1.0 + c)
        dN1db = -0.125 * (1.0 - a) * (1.0 - c)
        dN2db = -0.125 * (1.0 + a) * (1.0 - c)
        dN3db = 0.125 * (1.0 + a) * (1.0 - c)
        dN4db = 0.125 * (1.0 - a) * (1.0 - c)
        dN5db = -0.125 * (1.0 - a) * (1.0 + c)
        dN6db = -0.125 * (1.0 + a) * (1.0 + c)
        dN7db = 0.125 * (1.0 + a) * (1.0 + c)
        dN8db = 0.125 * (1.0 - a) * (1.0 + c)
        dN1dc = -0.125 * (1.0 - a) * (1.0 - b)
        dN2dc = -0.125 * (1.0 + a) * (1.0 - b)
        dN3dc = -0.125 * (1.0 + a) * (1.0 + b)
        dN4dc = -0.125 * (1.0 - a) * (1.0 + b)
        dN5dc = 0.125 * (1.0 - a) * (1.0 - b)
        dN6dc = 0.125 * (1.0 + a) * (1.0 - b)
        dN7dc = 0.125 * (1.0 + a) * (1.0 + b)
        dN8dc = 0.125 * (1.0 - a) * (1.0 + b)
        # dNdaを計算する
        dNda = jnp.array([
            [dN1da, dN2da, dN3da, dN4da, dN5da, dN6da, dN7da, dN8da],
            [dN1db, dN2db, dN3db, dN4db, dN5db, dN6db, dN7db, dN8db],
            [dN1dc, dN2dc, dN3dc, dN4dc, dN5dc, dN6dc, dN7dc, dN8dc]
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
