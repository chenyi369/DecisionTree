from math import log
import pandas as pd
import numpy as np

def read_xlsx(csv_path):
    data = pd.read_excel(csv_path)
    print(data)
    return data

def train_test_split(data, test_size=0.2, random_state=None):
    index = data.shape[0]
    # 设置随机种子，当随机种子非空时，将锁定随机数
    if random_state:
        np.random.seed(random_state)
        # 将样本集的索引值进行随机打乱
        # permutation随机生成0-len(data)随机序列
    shuffle_indexs = np.random.permutation(index)
    # 提取位于样本集中20%的那个索引值
    test_size = int(index * test_size)
    # 将随机打乱的20%的索引值赋值给测试索引
    test_indexs = shuffle_indexs[:test_size]
    # 将随机打乱的80%的索引值赋值给训练索引
    train_indexs = shuffle_indexs[test_size:]
    # 根据索引提取训练集和测试集
    train = data.iloc[train_indexs]
    test = data.iloc[test_indexs]
    return train, test


def calculateshang(data):
    y = data[data.columns[-1]]  #依据公式求某列特征的熵 目标变量作为概率依据
    n = len(y)
    labels = {}
    for i, j in y.value_counts().items():
        labels[i] = j
        # print(labels)
    shang = 0
    for i in labels:            #利用循环求熵
        pi = labels[i]/n
        shang -= pi * log(pi, 2)
    return shang

def choose(data):
    features = data.columns[:-1]
    ginis = []
    shang = calculateshang(data)
    for feature in features:
        li = pd.unique(data[feature])
        len = data.shape[0]
        tiaojianshang = 0
        for i in li:
            df = data[data[feature] == i]
            n = df.shape[0]
            pi = n/len
            tiaojianshang += pi * calculateshang(df)
            # print(tiaojianshang)
        gini = shang - tiaojianshang
        ginis.append(gini)
    ginis = np.array(ginis)
    a = np.argsort(-ginis)
    bestfeature = features[a[0]]
    # print(bestfeature)
    return bestfeature

def splitdataSet(data,feature,value):
    df = data[data[feature] == value]
    df = df.drop(feature, axis=1)
    return df


def major_k(classlist):
    classcount = classlist.value_counts()
    result = classcount.sort_values(ascending=False).index[0]
    return result

def createtree(data):
    labels = data.columns
    classlist = data[data.columns[-1]]
    if(classlist.value_counts().shape[0]==1):   #结束条件1：该分支目标变量唯一
        return classlist.values[0]
    if(len(labels) == 1):                          #结束条件2：所有特征名都循环完了
        return major_k(classlist)   #这里并不能直接返回目标变量名，可能不唯一，所以调用major_k
    bestFeature = choose(data)
    myTree = {bestFeature:{}}   #这里很巧妙，以此来创建嵌套字典
    unique = data[bestFeature].unique()
    for i in range(len(unique)):
        value = unique[i]
        myTree[bestFeature][value] = createtree(splitdataSet(data,bestFeature,value))   #递归创建树
    return myTree

def classfiy(myTree,labels,test):
    firstStr = list(myTree.keys())[0]       #需要获取首个特征的列号，以便从测试数据中取值比较
    secondDict = myTree[firstStr]           #获得第二个字典
    featIndex = labels.index(firstStr)      #获取测试集对应特征数值
    for key in secondDict.keys():
        if(test[featIndex] == key):
            if(type(secondDict[key]).__name__ == 'dict'):       #判断该值是否还是字典，如果是，则继续递归
                classlabel = classfiy(secondDict[key],labels,test)
            else:
                classlabel = secondDict[key]
    # print(classlabel)
    return classlabel

def accuracy(train,test):
    myTree = createtree(train)
    labels = list(test.columns)
    correct = 0
    for i in range(len(test)):
        testvalue = test.iloc[[i],:-1]
        classlabel = classfiy(myTree,labels,testvalue.values[0])
        if test.iloc[[i],-1].values[0] == classlabel:
            correct +=1
        # print(classlabel)
    accuracy = (correct / float(len(test))) * 100.0
    print("Accuracy:",accuracy,"%")
    return accuracy



if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\daikuan.xlsx')
    print(data)
    train, test = train_test_split(data)
    bestfeature = choose(data)
    # print(bestfeature)
    myTree = createtree(data)
    print(myTree)
    accuracy(train, test)
    # print(test.iloc[[0],-1].values[0])
    # classfiy(myTree, list(test.columns), testvalue)





