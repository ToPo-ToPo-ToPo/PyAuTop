
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import abc

#=============================================================================
# 要素クラスのインターフェースを定義する
# 具体的な実装は継承先にて行う
#=============================================================================
class ElementInterface(metaclass=abc.ABCMeta):
    
    #---------------------------------------------------------------------
    # 要素接線剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    @abc.abstractmethod
    def make_K(self):
        raise NotImplementedError()
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    @abc.abstractmethod
    def make_Fint(self):
        raise NotImplementedError()
    
    #---------------------------------------------------------------------
    # 等価節点力の荷重ベクトルを作成する
    #---------------------------------------------------------------------
    @abc.abstractmethod
    def make_Fb(self):
        raise NotImplementedError()
    