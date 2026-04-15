import os
# BPE构建词 Byte Pair Encoding 数据压缩、子词分词

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    i = 0
    new_ids = []
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def build_vocab(text):
    vocab_size = 500
    num_merges = vocab_size - 256
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging{pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx #合并的词的字典
    vocab = {idx: bytes([idx]) for idx in range(256)}  #初始化一个包含所有可能字节值的词汇表‌。

    for (p0,p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        try:
            print(idx, vocab[idx].decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return merges, vocab

def decode(ids, vocab):
    tokens =b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) > 2:
        stats = get_stats(tokens)
        pair = min(stats, key = lambda p : merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

if __name__ == "__main__":
    dir_path = r"G:\AI\每周的课\1.25第十四周 大语言模型应用相关\week14 大语言模型应用相关\RAG\dota2英雄介绍-byRAG\Heroes"
    #所有文件读成一个长字符串。也可以试试只读入一个文件
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'
    #构建词表
    merges, vocabs = build_vocab(corpus)
    #使用词表进行编解码
    string = "矮人直升机"
    encode_ids = encode(string, merges)
    print("编码结果：", encode_ids)
    decode_string = decode(encode_ids, vocabs)
    print("解码结果：", decode_string)
