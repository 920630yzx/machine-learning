from datetime import datetime
import pandas as pd
import portfolioopt as opt
symbol=['600036.XSHG','600050.XSHG','601318.XSHG'] #定义列表
data_dict = {} #定义字典
for s in symbol:  #循环读取,这样字典就可以存放3个股票数据('600036.XSHG','600050.XSHG','601318.XSHG')
    data =  pd.read_excel('sz50.xlsx',sheetname=s, index_col='datetime')
    data_dict[s] = data
PN = pd.Panel(data_dict) #将字典转换成Panel格式
print(PN)
data_r = PN.minor_xs('close').pct_change()[1:] #minor_xs表示minor_axis轴，得到收盘价每日收益率
data_1=PN.minor_xs('close') #这里得到每日收盘价格
print (data_r.head())

#求出年收益及其年收益率的协方差
exp_rets = data_r.mean()*252 #data_r.mean()表示求均值，在这里是求年收益(对比下精度如何)
cov_mat = data_r.cov()*252 #这里求年收益率的协方差
exp=(data_1.iloc[-1]-data_1.iloc[0])/data_1.iloc[0] #这里进行一个对比,发现还是有点差距
print (exp_rets)
print (cov_mat)



#计算：
#计算目标收益的权重 (markowitz_portfolio方法)
portfolio_1 = opt.markowitz_portfolio(cov_mat, exp_rets, 0.2, allow_short=False, market_neutral=False)
#需输入协方差矩阵cov_mat,年预期收益exp_rets,0.2代表想要的年收益,allow_short表示是否允许做空,market_neutral表示是否具有市场中性
print(portfolio_1)
#得到的结果表示若要实现0.2的年收益,则分别需买入这些股票的比重分别为

#计算最小方差的权重 (opt.min_var_portfolio)
portfolio_mv = opt.min_var_portfolio(cov_mat, allow_short=False)
print(portfolio_mv)

#计算最优组合的权重 (opt.tangency_portfolio) (夏普比率最高的比重)
portfolio_tp = opt.tangency_portfolio(cov_mat, exp_rets, allow_short=False) #需输入协方差矩阵cov_mat,年预期收益exp_rets
print(portfolio_tp)

#去除少于0.01权重的股票,低于0.01权重的不建议购买
weigth_t = opt.truncate_weights(portfolio_tp, min_weight=0.01, rescale=True)
print(weigth_t)



#计算组合风险 
import numpy as np
Portfolio_v = np.dot(weigth_t.T, np.dot(cov_mat,weigth_t)) #weigth_t.T表示weigth_t的转置,cov_mat是协方差矩阵
P_sigma = np.sqrt(Portfolio_v) #开方求标准差
P_sigma



#Markowitz可视化 (求最高夏普比率) ？？？
import matplotlib.pyplot as plt
port_returns = []
port_variance = []
for p in range(4000): #随机4000个值
    weights = np.random.rand(len(data_r.T)) #np.random.rand(3)#生成一个形状为3*1的一维数组,每个数字在0-1之间
    weights /= np.sum(weights) #求weights各项占比
    port_returns.append(np.sum(data_r.mean()*252*weights)) #计算年总收益率
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(cov_mat,weights)))) #np.sqrt(np.dot(weights.T, np.dot(cov_mat,weights)))计算组合风险
port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

risk_free = 0.04
P_r = np.sum(exp_rets*weigth_t)  #exp_rets*weigth_t对应求积
sharpe = (P_r-risk_free)/P_sigma  #夏普比率
print('sharpe:', sharpe)  #打印夏普比率

plt.figure(figsize = (15,7))
plt.scatter(port_variance, port_returns, c=(port_returns-risk_free)/port_variance, marker='o')
#port_variance表示风险,port_returns表示收益,
plt.grid(True)
plt.xlabel('excepted volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.scatter(P_sigma, P_r, c='r', marker='*') #plt.scatter表示画点
plt.show()








