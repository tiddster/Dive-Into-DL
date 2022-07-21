import pandas as pd
import torch
import torch.utils.data as torchData
import DIDLutils

"""
读取数据
"""
train_path = "datasetOfHousePrice\\train.csv"
test_path = "datasetOfHousePrice\\test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_feature = train_data.iloc[:, 1:-1]
test_feature = test_data.iloc[:, 1:]
y = train_data.iloc[:, -1:]

all_feature = pd.concat((train_feature, test_feature))

"""
数据清洗：数据中有很多str类型数值，需要将其转换为离散数据(0,1,2..)，以方便模型的训练
"""
# 挑选出数字型数据特征
numeric_feature_index = all_feature.dtypes[all_feature.dtypes != 'object'].index
# 将数字型数字特征标准化
all_feature[numeric_feature_index] = all_feature[numeric_feature_index].apply(lambda x: (x - x.mean()) / x.std())
# 将数字型数字特征中的缺失值用0填充
all_feature[numeric_feature_index] = all_feature[numeric_feature_index].fillna(0)

# 接下来将离散数值转成指示特征, 举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM,
# 那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
# 如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
all_feature = pd.get_dummies(all_feature, dummy_na=True)

# 第17行将所有特征合并进行数据清晰，现在将其分开，并由dataFrame格式转换为numpy格式，再转换成tensor格式
train_feature = torch.from_numpy(all_feature.iloc[:train_data.shape[0], :].to_numpy()).float()
test_feature = torch.from_numpy(all_feature.iloc[train_data.shape[0]:, :].to_numpy()).float()
train_labels = torch.from_numpy(y.to_numpy()).float()
print(train_labels.shape)

num_data, num_feature = train_feature.shape

"""
定义模型
"""
loss = torch.nn.MSELoss()

net = torch.nn.Sequential(
    torch.nn.Linear(num_feature, 1)
)

for param in net.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)


# 对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


"""
训练模型
"""


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_loss, test_loss = [], []

    dataset = torchData.TensorDataset(train_features, train_labels)
    train_iter = torchData.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_loss.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels))
    return train_loss, test_loss


"""
k折交叉验证集:它将被用来选择模型设计并调节超参数。DIDLutils中实现了一个函数，它返回第i折交叉验证时所需要的训练和验证数据。
"""
DIDLutils.k_fold(net, train, 5, train_feature, train_labels, num_epochs=100, learning_rate=5, weight_decay=0, batch_size=50)
