#! /usr/bin/env python
# -*-coding=utf8-*-

# metaclass=Singleton
class PS:
    def __init__(self, embedding_dim):
        np.random.seed(2020)
        self.params_server = dict()
        self.dim = embedding_dim
        print("ps inited...")
        
    def pull(self, keys):  # 从参数服务器拉去特征所对应的参数
        values = []
        # 这里传进来的数据是[batch, feature_len]->一个样本的数据，样本的特征长度
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)
        
        return np.asarray(values, dtype='float32')
    
    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):  # [batch, feature_len]
                self.params_server[keys[i][j]] = values[i][j]
    
    
    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)
            
    def save(self, path):
        print("总共包含keys： ", len(self.params_server))
        writer = open(path, "w")
        for k, v in self.params_server.items():
            writer.write(str(k) + "\t" + ",".join(["%.8f" % _ for _ in v]) + "\n")
        writer.close() 