
from functools import partial
import jax.numpy as jnp
from jax import jit
from linear_elastic_plane_stress import LinearElasticPlaneStress
from linear_elastic_brick import LinearElasticBrick
from boundary_condition import BoundaryConditions, DirichletBc, NeummanBc
import make_hexa8_voxel_model
from solid_mechanics import SolidMechanics
from linear_fem import LinearFEM
#=============================================================================
# メインプログラム
#=============================================================================
def main_test():
    
    #
    num_step = 1
    
    # 節点とコネクティビティを作成
    nodes, elements = make_hexa8_voxel_model.run(xlength=3.0, ylength=1.0, zlength=1.0, xdiv=3, ydiv=1, zdiv=1)
    
    # 材料物性を定義
    material = LinearElasticBrick(young=1.0, poisson=0.3, density=1.0e-05)
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
    range_min = jnp.array([0.0, 0.0, 0.0])
    range_max = jnp.array([0.0, 1.0, 1.0])
    flags = [1, 1, 1]
    u_bar = set_dirichlet_value(num_step=num_step)
    def find_dirichlet_nodes(nodes, range_min, range_max):
        bc_nodes = []
        delta = 1.0e-05
        for node in nodes:
            x, y, z = node.coordinate
            if (range_min[0]-delta < x < range_max[0]+delta) and (range_min[1]-delta < y < range_max[1]+delta) and (range_min[2]-delta < z < range_max[2]+delta):
                bc_nodes.append(node)
        return bc_nodes
    dirich_nodes = find_dirichlet_nodes(nodes, range_min=range_min, range_max=range_max)
    
    # 条件の追加
    boundary_conditions.dirichlet_bcs.append(DirichletBc(nodes=dirich_nodes, flags=flags, values=u_bar))
    
    # 境界条件の設定 ノイマン境界
    @partial(jit, static_argnums=(0))
    def set_neumman_value(num_step):
        t_bar = jnp.empty((num_step, 3))
        for istep in range(num_step):
            t_bar = t_bar.at[istep, 0].set(0.0)
            t_bar = t_bar.at[istep, 1].set(-1.0)
        return t_bar
    
    # 条件を設定
    range_min = jnp.array([3.0, 0.0, 0.0])
    range_max = jnp.array([3.0, 1.0, 1.0])
    t_bar = set_neumman_value(num_step=num_step)
    
    def find_neumman_nodes(nodes, range_min, range_max):
        bc_nodes = []
        delta = 1.0e-05
        for node in nodes:
            x, y, z = node.coordinate
            if (range_min[0]-delta < x < range_max[0]+delta) and (range_min[1]-delta < y < range_max[1]+delta) and (range_min[2]-delta < z < range_max[2]+delta):
                bc_nodes.append(node)
        return bc_nodes
    neum_nodes = find_neumman_nodes(nodes=nodes, range_min=range_min, range_max=range_max)
    
    # 条件の追加
    boundary_conditions.neumman_bcs.append(NeummanBc(nodes=neum_nodes, values=t_bar))
                
    # 解析モデルの作成
    model = SolidMechanics(nodes=nodes, elements=elements, boundary_conditions=boundary_conditions)
    
    # 解析手法の設定
    method = LinearFEM(model=model, num_step=num_step)
    
    # 解析の実行
    U_list, Freat_list = method.run()
    
    # ポスト処理
    print(U_list)
    
    
#=============================================================================
# メインプログラム
#=============================================================================
main_test()