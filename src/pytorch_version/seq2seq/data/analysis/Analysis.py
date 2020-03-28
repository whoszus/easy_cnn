import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn



def load_data_by_date(df, start, end):
    mask = (df['alm_time'] >= start) & (df['alm_time'] <= end)
    return df.loc[mask]


def load_data(start='2018-06-01', end='2018-06-02', path='ays.torch'):
    if not os.path.exists(path):
        return
    data_load = torch.load(path)['data_all']
    start_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')
    data_loaded = load_data_by_date(data_load, start_time, end_time)

    return data_loaded


# 分析恢复时间与告警级别关系
def analysis_clrtime_and_level():
    if os.path.exists("analysis_test.torch"):
        data = torch.load("analysis_test.torch")['data']
    else:
        data = load_data()
        data = data.sort_values("alm_time")
        save_file = {
            'data': data
        }
        torch.save(save_file, "analysis_test.torch")
    data_with_r = data.loc[(data.clr_time.notnull())]
    data_with_r['RT'] =  data_with_r.apply(lambda x:x['clr_time']-x['alm_time'], axis=1)
    data_with_r['RT'] = data_with_r['RT'].dt.total_seconds()
    # ["city","dev_name","dev_type","spc_type","alm_time","clr_time","ralm_type","ralm_level","msg_type","RT"]


    data_with_r.ralm_level.value_counts().plot(kind='pie',)
    plt.savefig('pie_ralmLevel_count.jpg')

if __name__ == '__main__':

    analysis_clrtime_and_level()
    # plt.plot(range(20), range(20))
    # plt.interactive(True)
    # plt.savefig( 'myfig.png' )
