class Node1d:
    # コンストラクタ
    # no : 節点番号
    # x  : x座標
    # y  : y座標
    def __init__(self, no, x):
        self.no = no   # 節点番号
        self.x = x     # x座標

    # 節点の情報を表示する
    def printNode(self):
        print("Node No: %d, x: %f" % (self.no, self.x))
