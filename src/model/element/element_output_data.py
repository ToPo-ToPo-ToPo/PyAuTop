#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

# 要素の出力データを格納するクラス
class ElementOutputData:

    # コンストラクタ
    # element        : 要素のクラス
    # elem_solution    : 要素節点の変位のリスト
    # vecStressList  : 積分点の応力のリスト(np.array型のリスト)
    # vecEStrainList : 積分点の弾性ひずみのリスト(np.array型のリスト)
    # vecPStrainList : 積分点の塑性ひずみのリスト(np.array型のリスト)
    # ePStrainList   : 相当塑性ひずみのリスト
    # misesList      : ミーゼス応力のリスト
    def __init__(self, element, vecStressList, vecEStrainList, vecPStrainList, ePStrainList, misesList):

        # インスタンス変数を定義する
        self.element = element

        # 積分点の応力、ひずみを計算する
        self.vecIpPStrainList = vecPStrainList                  # 積分点の塑性ひずみ
        self.vecIpStrainList = []                               # 積分点のひずみベクトルのリスト(np.arrayのリスト型)
        
        for i in range(len(vecEStrainList)):
            vecIpStrain = vecEStrainList[i] + vecPStrainList[i]
            self.vecIpStrainList.append(vecIpStrain) 
        
        self.ipEPStrainList = ePStrainList                      # 積分点の相当塑性ひずみ
        self.vecIpStressList = vecStressList                    # 積分点の応力ベクトルのリスト(np.arrayのリスト型)
        self.ipMiseses = misesList                              # 積分点のミーゼス応力
