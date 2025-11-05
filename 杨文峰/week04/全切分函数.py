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
#待切分文本
sentence = "经常有意见分歧"
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    def back_f(start,path):
        if start == len(sentence):
            target.append(path[:]) #浅拷贝
            return

        for end in range(start + 1, len(sentence) + 1):
            if sentence[start:end] in Dict:
                path.append(sentence[start:end])
                back_f(end, path)
                path.pop()

    back_f(0, [])
    return target

if __name__ == "__main__":
    target = all_cut(sentence, Dict)
    print(target)
