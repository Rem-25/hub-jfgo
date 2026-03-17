Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

def dag(sentence):
    DAG = {}
    length = len(sentence)
    for k in range(length):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < length:
            if frag in Dict:
                tmplist.append(i)
            i += 1
            frag = sentence[k : i+1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG

class DAG_mod:
    def __init__(self,sentence):
        self.sentence = sentence
        self.DAG = dag(sentence)
        self.unfinish_path = [[]]
        self.finish_path = []
        self.length = len(sentence)

    def decode_first(self, path):
        path_length = len("".join(path))
        if path_length == self.length:
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_path = []
        for candidate in candidates:
            new_path.append(path + [self.sentence[path_length : candidate + 1]])
        self.unfinish_path = self.unfinish_path + new_path
        return

    def decode_last(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()
            self.decode_first(path)

sentence = "经常有意见分歧"
DAG_D = DAG_mod(sentence)
DAG_D.decode_last()
print(DAG_D.finish_path)

