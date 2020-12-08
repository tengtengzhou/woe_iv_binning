import pandas as pd
import numpy as np

class chi_binning_woe_iv:
    def __init__(self,
                 confidenceVal=3.841,
                 bin=10,
                 sample=None,
                 lang='en',
                 method='chi2',
                 limit_ratio=0.05):
        self.confidenceVal = confidenceVal
        self.bin = bin
        self.sample = sample
        self.lang = lang
        self.method = method
        self.limit_ratio = limit_ratio

    def fit(self, df, variable, target):
        self.df = df
        self.variable = variable
        self.target = target
        if self.method == 'chi2':
            bins = self.fit_helper1_chi_binning(self.df, self.variable,
                                                self.target,
                                                self.confidenceVal, self.bin,
                                                self.sample)
        elif self.method == 'best_ks':
            bins = self.fit_helper1_ks_binning(self.df, self.variable,
                                               self.target, self.bin,
                                               self.limit_ratio)
        elif self.method == 'qcut':
            bins = self.fit_helper1_qcut_binning(self.df,self.variable,self.target,self.bin)
        else:
            return 'wrong binning method'
        #print(bins)
        new_col, iv = self.fit_helper2_woe_change(self.df, self.variable, bins)
        self.iv = iv

    def fit_helper1_chi_binning(self,
                                df,
                                variable,
                                flag,
                                confidenceVal=3.841,
                                bin=10,
                                sample=None):
	        '''
	        运行前需要 import pandas as pd 和 import numpy as np
	        df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
	        variable:需要卡方分箱的变量名称（字符串）
	        flag：正负样本标识的名称（字符串）
	        confidenceVal：置信度水平（默认是不进行抽样95%）
	        bin：最多箱的数目
	        sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
	        '''
        #进行是否抽样操作
        if sample != None:
            df = df.sample(n=sample)
        else:
            df

        #进行数据格式化录入
        total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
        total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
        positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
        positive_class = pd.DataFrame({'positive_class':
                                       positive_class})  # 创建一个数据框保存之前的结果
        regroup = pd.merge(total_num,
                           positive_class,
                           left_index=True,
                           right_index=True,
                           how='inner')  # 组合total_num与positive_class
        regroup.reset_index(inplace=True)
        regroup['negative_class'] = regroup['total_num'] - regroup[
            'positive_class']  # 统计需分箱变量每个值负样本数
        regroup = regroup.drop('total_num', axis=1)
        np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
        #print('已完成数据读入,正在计算数据初处理')
        #print(regroup)

        #处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
        i = 0
        while (i <= np_regroup.shape[0] - 2):
            if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0)
                    or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
                np_regroup[i,
                           1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
                np_regroup[i,
                           2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
                np_regroup[i, 0] = np_regroup[i + 1, 0]
                np_regroup = np.delete(np_regroup, i + 1, 0)
                i = i - 1
            i = i + 1
        #print(np_regroup)

        #对相邻两个区间进行卡方值计算
        chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
        for i in np.arange(np_regroup.shape[0] - 1):
            chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
              * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
              np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
            chi_table = np.append(chi_table, chi)
        #print('已完成数据初处理，正在进行卡方分箱核心操作')
        #print(chi_table)

        #把卡方值最小的两个区间进行合并（卡方分箱核心）
        while (1):
            if (len(chi_table) <= (bin - 1)
                    and min(chi_table) >= confidenceVal):
                break
            chi_min_index = np.argwhere(
                chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
            np_regroup[chi_min_index,
                       1] = np_regroup[chi_min_index,
                                       1] + np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index,
                       2] = np_regroup[chi_min_index,
                                       2] + np_regroup[chi_min_index + 1, 2]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                               * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)

            else:
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
                # 计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                           * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
        #print('已完成卡方分箱核心操作，正在保存结果')
        #print(chi_table)

        #把结果保存成一个数据框
        result_data = pd.DataFrame()  # 创建一个保存结果的数据框
        result_data['variable'] = [variable
                                   ] * np_regroup.shape[0]  # 结果表第一列：变量名
        list_temp = []
        left_bnd = []
        right_bnd = []
        for i in np.arange(np_regroup.shape[0]):
            if i == 0:
                left = -float('inf')
                right = np_regroup[i, 0]
            elif i == np_regroup.shape[0] - 1:
                left = np_regroup[i - 1, 0]
                right = float('inf')
            else:
                left = np_regroup[i - 1, 0]
                right = np_regroup[i, 0]
            x = '(' + str(left) + ', ' + str(right) + ']'
            list_temp.append(x)
            left_bnd.append(left)
            right_bnd.append(right)

        result_data['interval'] = list_temp  # 结果表第二列：区间
        result_data['left_bnd'] = left_bnd
        result_data['right_bnd'] = right_bnd
        result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
        result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
        result_data['flag_0_ratio'] = result_data['flag_0'] / sum(
            result_data['flag_0'])  # 结果表第5列：负样本的总体比例
        result_data['flag_1_ratio'] = result_data['flag_1'] / sum(
            result_data['flag_1'])  # 结果表第6列：正样本的总体比例

        return result_data

    def fit_helper1_ks_binning(self,
                               df,
                               variable,
                               flag,
                               bin=10,
                               limit_ratio=0.05):
        sep_points = self.fit_helper1_ks_binning_get_sep_points(
            df, variable, flag, bin=10, limit_ratio=0.05)
        data_table = self.fit_helper1_1_from_sep_points_to_tables(
            df, variable, flag, sep_points)
        return data_table

    def fit_helper1_ks_binning_get_sep_points(self,
                                              df,
                                              variable,
                                              flag,
                                              bin=10,
                                              limit_ratio=0.05):
        cnt = 0
        data = df[[variable, flag]].copy()
        n = df.shape[0]
        all_sep_points = []
        stack = [data]
        while (stack and cnt <= bin):
            df0 = stack.pop(0)
            #print(cnt)
            #print(df0)
            df1 = pd.crosstab(df0['x'], df0['y'])
            all_x = np.array(df1.index)  # save all x values
            df1 = np.array(df1)
            df2 = df1.cumsum(0) / df1.sum(0)  # cal ks
            idx = np.argmax(df2[:, 0] - df2[:, 1])  # find best split idx
            sep_point = all_x[idx]
            #print(sep_point)
            all_sep_points.append(sep_point)

            data1 = df0[df0[variable] <= sep_point]
            data2 = df0[df0[variable] > sep_point]
            #print(len(data1[flag].unique()),len(data2[flag].unique()))
            if data1.shape[0] > limit_ratio * n and len(
                    data1[flag].unique()) == 2:
                stack.append(data1)
                cnt += 1
            if data2.shape[0] > limit_ratio * n and len(
                    data2[flag].unique()) == 2:
                stack.append(data2)
                cnt += 1
        return np.sort([-float('inf')] + list(set(all_sep_points)) +
                       [float('inf')])

    def fit_helper1_qcut_binning(self,df,variable,flag,q_num):
        q_2 = [100/q_num*i for i in range(1,q_num)]
        lst1 = list(set([np.percentile(temp['x'],i) for i in q_2]))
        res = np.round(np.sort([float('inf'),-float('inf')] + lst1),8)
        return self.fit_helper1_1_from_sep_points_to_tables(df,variable, flag,res)
    
    def fit_helper1_1_from_sep_points_to_tables(self, df, variable, flag,
                                            sep_points):
        n = len(sep_points) - 1
        data_table = pd.DataFrame(np.zeros((n, 6)))
        data_table.columns = [
            variable, 'interval', 'left_bnd', 'right_bnd', 'flag_0', 'flag_1'
        ]
        data_table[variable] = variable
        for i in range(n):
            left = sep_points[i]
            right = sep_points[i + 1]
            data_table.iloc[i, 1] = '(' + str(left) + ',' + str(right) + ']'
            data_table.iloc[i, 2] = left
            data_table.iloc[i, 3] = right
            tmp = df[(df[variable] > left) & (df[variable] <= right)]
            flag_0 = sum(tmp[flag] == 0)
            flag_1 = sum(tmp[flag] == 1)
            data_table.iloc[i, 4] = flag_0
            data_table.iloc[i, 5] = flag_1
        data_table['flag_0_ratio'] = data_table['flag_0'] / sum(
            data_table['flag_0'])
        data_table['flag_1_ratio'] = data_table['flag_1'] / sum(
            data_table['flag_1'])
        return data_table

    def fit_helper2_woe_change(self,
                               df,
                               variable,
                               results_data,
                               inplace=False):
        """
        Argument:
            df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
            variable:需要卡方分箱的变量名称（字符串）
            results_data:由ChiMerge产生的表
            inplace: if True直接在temp上修改成分箱后的数据
        Return:
            new_col_df['new_col']:woe转化后的pd.series格式的数据
            sum(results_data['iv']):这个变量的iv值
        """
        results_data['BAD_RATIO'] = results_data['flag_1'] / (
            results_data['flag_1'] + results_data['flag_0'])
        results_data['flag_0_fixed'] = results_data['flag_0'].replace(0, 1)
        results_data['flag_1_fixed'] = results_data['flag_1'].replace(0, 1)
        results_data['flag_0_fixed_ratio'] = results_data[
            'flag_0_fixed'] / sum(results_data['flag_0_fixed'])
        results_data['flag_1_fixed_ratio'] = results_data[
            'flag_1_fixed'] / sum(results_data['flag_1_fixed'])
        results_data['BAD_RATIO_FIXED'] = results_data['flag_1_fixed'] / (
            results_data['flag_1_fixed'] + results_data['flag_0_fixed'])
        results_data['woe'] = np.log(results_data['flag_1_fixed_ratio'] /
                                     results_data['flag_0_fixed_ratio'])
        results_data['iv'] = (results_data['flag_1_fixed_ratio'] - \
                              results_data['flag_0_fixed_ratio'])*results_data['woe']

        new_col_df = df[[variable]].copy()
        new_col_df['new_col'] = new_col_df[variable]
        for i in range(results_data.shape[0]):
            new_col_df['new_col'][
                (results_data['left_bnd'][i] < new_col_df[variable])
                & (new_col_df[variable] <= results_data['right_bnd'][i]
                   )] = results_data['woe'][i]
        #print(results_data)
        if inplace == True:
            df[variable] = new_col_df['new_col']

        self.results_data = results_data

        return new_col_df['new_col'], sum(results_data['iv'])

    def transform(self, df):
        variable = self.variable
        results_data = self.results_data

        new_col_df = df[[variable]].copy()
        new_col_df['new_col'] = new_col_df[variable]
        for i in range(results_data.shape[0]):
            new_col_df['new_col'][
                (results_data['left_bnd'][i] < new_col_df[variable])
                & (new_col_df[variable] <= results_data['right_bnd'][i]
                   )] = results_data['woe'][i]
        res = new_col_df['new_col']
        res = np.array(res)
        return res

    def get_iv_value(self, ):
        return self.iv

    def get_woe_table(self, n_rounds=None):
        if self.lang != 'en':
            new_col_name = [
                '变量名称', '区间', '左边界', '右边界', '好人数量', '坏人数量', '好人的总体占比',
                '坏人的总体占比', '坏账率_y', '好人数量_修正后', '坏人数量_修正后', '好人的总体占比_修正后',
                '坏人的总体占比_修正后', '坏账率_y_修正后', 'woe', 'iv'
            ]
            #print(self.results_data.columns)
            #print(new_col_name)
            self.results_data.columns = new_col_name
        if n_rounds:
            return round(self.results_data, n_rounds)
        return self.results_data

####Example data
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_iris

#iris = load_iris()
#data = pd.DataFrame(iris['data'])
#data['y'] = iris['target']
#temp = data.iloc[:,[0,4]]
#temp.columns = ['x','y']
#temp = temp[temp['y']!=2]
#temp_train, temp_test = train_test_split(temp,test_size = 0.3, random_state = 454564789)


####eg1: chi2 binning 
	#chi_bin1 = chi_binning_woe_iv(confidenceVal=2, bin=10, sample = None,lang = 'cn')
	#chi_bin1.fit(temp_train,'x','y')
	#x_test_trasformed = chi_bin1.transform(temp_test)
	#x_test_trasformed
	#chi_bin1.get_iv_value()
	#chi_bin1.get_woe_table()

####eg2: best_ks binning
	#ks_bin1 = chi_binning_woe_iv(bin=10, sample = None,lang = 'cn',method='best_ks',
	#                           limit_ratio=0.05)
	#ks_bin1.fit(temp_train,'x','y')
	#x_test_trasformed = ks_bin1.transform(temp_test)
	#x_test_trasformed
	#ks_bin1.get_iv_value()
	#ks_bin1.get_woe_table(n_rounds=4)

####eg3: qcut binning
	#qcut_bin1 = chi_binning_woe_iv(bin=5, sample = None,lang = 'cn',method='qcut')
	#qcut_bin1.fit(temp_train,'x','y')
	#x_test_trasformed = qcut_bin1.transform(temp_test)
	#x_test_trasformed
	#qcut_bin1.get_iv_value()
	#qcut_bin1.get_woe_table(n_rounds=4)












