
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import  value_and_grad
from linear_elastic_plane_stress import LinearElasticPlaneStress
from boundary_condition import BoundaryConditions, DirichletBc, NeummanBc
import make_quad4_voxel_model
from solid_mechanics import SolidMechanics
from linear_fem import LinearFEM
#----------------------------------------------------------
# コンプライアンスを計算するクラス
#----------------------------------------------------------
class StaticStructuralCompliance:
    # 初期化
    def __init__(self, id, model, method):
        self.id = id
        self.model = model
        self.method = method
        self.function_controller = value_and_grad(self.compute)
    
    # コンプライアンスを計算
    @partial(jit, static_argnums=(0))   
    def compute(self, x):
                
        # 解析の実行
        U_list, Frac_list = self.method.run(x)

        # 外力の計算
        Fext = self.model.make_Ft(self.method.num_step)
        
        # 変位の取得
        U = U_list[self.method.num_step, :]
    
        # 関数の計算
        comp = jnp.dot(Fext, U)
        return comp

#=============================================================================
# メインプログラム
#=============================================================================
def main_test():
    
    #
    num_step = 1
    
    # 節点とコネクティビティを作成
    nodes, elements = make_quad4_voxel_model.run(xlength=3.0, ylength=1.0, xdiv=10, ydiv=5)
    
    # 材料物性を定義
    material = LinearElasticPlaneStress(young=1.0, poisson=0.3, density=1.0e-05)
    for element in elements:
        element.set_material(material)
        
    # 境界条件の定義
    boundary_conditions = BoundaryConditions()
    
    # 境界条件の設定 ディレクレ境界
    @partial(jit, static_argnums=(0))
    def set_dirichlet_value(num_step):
        u_bar = jnp.empty((num_step, 3))
        for istep in range(num_step):
            for idof in range(3):
                # jaxを使用する際の配列の変更 (u_bar[istep, idof] = 0.0)
                u_bar = u_bar.at[istep, idof].set(0.0)
        return u_bar
    
    # 条件を設定
    range_min = jnp.array([0.0, 0.0])
    range_max = jnp.array([0.0, 1.0])
    flags = jnp.array([1.0, 1.0])
    u_bar = set_dirichlet_value(num_step=num_step)
    def find_dirichlet_nodes(nodes, range_min, range_max):
        bc_nodes = []
        delta = 1.0e-05
        for node in nodes:
            x, y, z = node.coordinate
            if (range_min[0]-delta < x < range_max[0]+delta) and (range_min[1]-delta < y < range_max[1]+delta):
                bc_nodes.append(node)
        return bc_nodes
    dirich_nodes = find_dirichlet_nodes(nodes, range_min=range_min, range_max=range_max)
    
    # 条件の追加
    boundary_conditions.dirichlet_bcs.append(DirichletBc(nodes=dirich_nodes, flags=flags, values=u_bar))
    
    # 境界条件の設定 ノイマン境界
    @partial(jit, static_argnums=(0))
    def set_neumman_value(num_step):
        t_bar = jnp.empty((num_step, 2))
        for istep in range(num_step):
            t_bar = t_bar.at[istep, 0].set(0.0)
            t_bar = t_bar.at[istep, 1].set(-1.0)
        return t_bar
    
    # 条件を設定
    range_min = jnp.array([3.0, 0.0])
    range_max = jnp.array([3.0, 1.0])
    t_bar = set_neumman_value(num_step=num_step)
    
    def find_neumman_nodes(nodes, range_min, range_max):
        bc_nodes = []
        delta = 1.0e-05
        for node in nodes:
            x, y, z = node.coordinate
            if (range_min[0]-delta < x < range_max[0]+delta) and (range_min[1]-delta < y < range_max[1]+delta):
                bc_nodes.append(node)
        return bc_nodes
    neum_nodes = find_neumman_nodes(nodes=nodes, range_min=range_min, range_max=range_max)
    
    # 条件の追加
    boundary_conditions.neumman_bcs.append(NeummanBc(nodes=neum_nodes, values=t_bar))
                
    # 解析モデルの作成
    model = SolidMechanics(nodes=nodes, elements=elements, boundary_conditions=boundary_conditions)
    
    # 解析手法の設定
    method = LinearFEM(model=model, num_step=num_step)

    # 評価関数の設定
    eval_function = StaticStructuralCompliance(id=1, model=model, method=method)

    # 設計変数の定義
    s = np.ones(len(model.elements))

    # 微分の計算
    value, df = eval_function.function_controller(s)
    print("評価関数値: ")
    print(value)
    
    print("感度値: ")
    print(df)

    # 設計変数の定義
    s[0] = 0.3

    # 微分の計算
    value, df = eval_function.function_controller(s)
    print("評価関数値: ")
    print(value)
    
    print("感度値: ")
    print(df)
    
    
#=============================================================================
# メインプログラム
#=============================================================================
main_test()
       