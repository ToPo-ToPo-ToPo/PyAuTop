
#=============================================================================
#
#=============================================================================
class Analysis:
    
    def __init__(self, id, physics, method, num_step):
        
        self.id = id
        self.physics = physics
        self.method = method
        self.num_step = num_step
    
    #---------------------------------------------------------------------
    # 解析を実行する
    #---------------------------------------------------------------------
    def run(self):
        
        #for istep in range(self.num_step):
        #    a = 1
        self.method.run()