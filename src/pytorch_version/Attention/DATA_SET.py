class M_Test_data():
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self, m_data):
        train_data_x, train_data_y = m_data
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
