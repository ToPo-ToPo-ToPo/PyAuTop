
from functools import partial
import jax.numpy as jnp
from jax import jit
from linear_elastic_plane_stress import LinearElasticPlaneStress
from linear_elastic_brick import LinearElasticBrick
import make_hexa8_voxel_model
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
    range_max = jnp.array([0.0, 0.0, 0.0])
    flags = [1, 1, 1]
    ubar = set_dirichlet_value(num_step=num_step)
    @partial(jit, static_argnums=(0, 1, 2, 3, 4))
    def find_dirichlet_nodeids(nodes, range_min, range_max, flags, ubar):
        node_ids = []
        for node in nodes:
            x, y, z = node.coordinate
            if (range_min[0] < x < range_max[0]) and (range_min[1] < y < range_max[1]) and (range_min[2] < z < range_max[2]):
                node_ids.append(node.id)
        
        node_ids_jnp = jnp.array(node_ids)
        return node_ids_jnp
                
    # 解析モデルの作成
    
    
    # 解析手法の設定
    
    
    # 解析の実行
    
    
    # ポスト処理
    
    
#=============================================================================
# メインプログラム
#=============================================================================
main_test()