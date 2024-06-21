import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
df_abtest = pd.read_csv('ab_test_click_data (1).csv')
# print(df_abtest.head())
# print(df_abtest.describe())
# print(df_abtest.groupby('group').sum('click'))
# palette = {0:'yellow',1:'black'}
# plt.figure(figsize = (10,6))
# ax = sns.countplot(x='group',hue='click',data=df_abtest,palette=palette)
# plt.title("Click Data Distribution in Experimental and Control Group")
# plt.xlabel("Group")
# plt.ylabel("Count")
# plt.legend(title='Click Data',labels=['No','Yes'])
# group_counts = df_abtest.groupby('group').size()
# group_click_counts = df_abtest.groupby(['group','click']).size().reset_index(name='clicks')
# for p in ax.patches:
#     height = p.get_height()
#     group = 'exp' if p.get_x() < 0.5 else 'con'
#     click = 1 if p.get_x() % 1 >0.5 else 0
#     total = group_counts.loc[group]
#     percentage = 100* height/total
#     ax.text(p.get_x()+p.get_width()/2.,height +5, f'{percentage:.1f}%',ha='center',color='black',fontsize=10)
# plt.tight_layout()
# plt.show()

alpha = 0.05
delta = 0.1
N_con = df_abtest[df_abtest["group"]=="con"].count()
N_exp = df_abtest[df_abtest["group"]=="exp"].count()
x_con = df_abtest.groupby('group')['click'].sum().loc['con']
x_exp = df_abtest.groupby('group')['click'].sum().loc['exp']
# print(df_abtest.groupby('group').sum('click'))
# print("number of users in control: ",N_con)
# print("number of users in exp: ",N_exp)
# print("number of clicks in control group: ", x_con)
# print("number of clicks in experimental group: ", x_exp)
# computing the estimate of click probability per group
p_con_hat = x_con/N_con   #clicks/impressions
p_exp_hat = x_exp/N_exp
print("Click Probability in Control Group:", p_con_hat)
print("Click Probability in Experimental Group:", p_exp_hat)
# computing the estimate of pooled clicked probability
p_pooled_hat = (x_con+x_exp)/(N_con + N_exp)
print("Pooled click probability: ", p_pooled_hat)
# computing the estimate of pooled variance
pooled_variance = p_pooled_hat * (1-p_pooled_hat) * (1/N_con + 1/N_exp)
print("p^_pooled is: ", p_pooled_hat)
print("pooled_variance is: ", pooled_variance)
# computing the standard error of the test
SE = np.sqrt(pooled_variance)
print("Standard Error is: ", SE)

# computing the test statistics of Z-test
Test_stat = (p_con_hat - p_exp_hat)/SE
print("Test Statistics for 2-sample Z-test is:", Test_stat)
Z_crit = norm.ppf(1-alpha/2)
print("Z-critical value from Standard Normal distribution: ", Z_crit)

p_value = 2 * norm.sf(abs(Test_stat))
def is_statistical_significance(p_value,alpha):
    print(f"P-value of 2 sample Z test: {np.around(p_value,3)}")
    if p_value.all()<alpha :
        print("Reject Null Hypothesis,means that the experimental group is statistical significance having positive effect in CTR")
    else:
        print("No difference between exp and con group, do not reject null")
is_statistical_significance(p_value,alpha)



# Parameters for the standard normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = norm.pdf(x, mu, sigma)

# Test statistics and critical value from previous calculation
Test_stat = -59.441633  # This value is from your A/B test results
Z_crit = 1.959963984540054  # Z-critical value for a 5% significance level in a two-tailed test

# # Plotting the standard normal distribution
# plt.plot(x, y, label='Standard Normal Distribution')
#
# # Shade the rejection region for a two-tailed test
# plt.fill_between(x, y, where=(x > Z_crit) | (x < -Z_crit), color='red', alpha=0.5, label='Rejection Region')
#
# # Adding Test Statistic
# plt.axvline(Test_stat, color='green', linestyle='dashed', linewidth=2, label=f'Test Statistic = {Test_stat:.2f}')
#
# # Adding Z-critical values
# plt.axvline(Z_crit, color='blue', linestyle='dashed', linewidth=1, label=f'Z-critical = {Z_crit:.2f}')
# plt.axvline(-Z_crit, color='blue', linestyle='dashed', linewidth=1)
#
# # Adding labels and title
# plt.xlabel('Z-value')
# plt.ylabel('Probability Density')
# plt.title('Gaussian Distribution with Rejection Region \n (A/B Testing for CTR button)')
# plt.legend()
#
# # Show plot
# plt.show()

CI = [np.around((p_exp_hat - p_con_hat) - SE*Z_crit,3), np.around((p_exp_hat - p_con_hat) + SE*Z_crit,3)]
print("Confidence Interval of the 2 sample Z-test is: ", CI)

def is_Practically_significance(delta1,CI_95):
    lower_bound_CI = CI_95[0]
    if delta1>=lower_bound_CI:
        print(f"We have practical significance \nWith MDE of {delta1}, the difference between con and exp group is significance")
        return True
    else:
        print("We dont have practical significance")
        return False
delta1 = 0.05
CI_95 =(0.04,0.06)
significance = is_Practically_significance(delta1,CI_95)
