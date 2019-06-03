![](https://upload-images.jianshu.io/upload_images/13575947-acc730f9a1734be9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Welcome to the real world

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description)是Kaggle的入门项目，本文就以它为例，介绍机器学习在真实的商业场景中是怎么玩的。Github: [here](https://github.com/alexshuang/house_price/blob/master/houseprice_rf.ipynb)。

## Look at the data

数据科学项目的一般流程是：收集数据，理解数据，数据清洗和特征工程，建立数据模型，调整模型参数来拟合数据。其中，理解数据是很关键的一步，我把它交给机器学习模型来完成，因为我相信机器更懂数据。

初步浏览完数据后，我就会用机器学习模型（Random Forest）筛选出对因变量（dependant variable）影响最大特征变量，重点分析这些特征，找出它们之间的因果和相关性。相比一上来就做全面的数据分析，这样的做法的效率会高很多。

之所以用Random Forest来筛选特征，最主要的原因是它的算法不仅简单高效，而且可解释。Random Forest是我常用的模型之一，它的优点很多，例如不挑数据、不需要做太多的特征工程、自带验证集（validation set）、不会大量过拟合等。

![](https://upload-images.jianshu.io/upload_images/13575947-c2cec0551512c312.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到，每个样本有81个特征，包括因变量（反应变量）-- SalePrice，和80个自变量。测试集除了没有因变量之外，其余的和训练集相同。

```
train_cats(train_df)
df, y, nas = proc_df(train_df, 'SalePrice')
yl = np.log(y)
m = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
%time m.fit(df, yl)
m.score(df, yl), m.oob_score_

CPU times: user 2.88 s, sys: 13.6 ms, total: 2.9 s
Wall time: 1.59 s
(0.9824407501222306, 0.8704055017750906)
```

train_cats()和proc_df()用于将非数值型的数据转化为数值（Numeralization）、填充NA值。经log转换后，y值从右偏转变成正态分布的图形，更利于模型训练。Random Forest的oob（out-of-bag）可以理解为模型自带的验证集，“oob_score=True”后就可以读出验证集的$R^2$ score，值越接近1表示模型效果越好。关于Random Forest更多的介绍可以看[【Predict Future Sales】玩转销量预测 part2](https://www.jianshu.com/p/1f6eef8a86fd)。

![](https://upload-images.jianshu.io/upload_images/13575947-3a4b96243cc929a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

RF认为最重要的前10个特征有：OverallQual、xxxArea、xxxSF、CentralAir、YearBuilt。

## Data Analysis
- **OverallQual: Overall material and finish quality**
![](https://upload-images.jianshu.io/upload_images/13575947-f51ec8e2cfe9b6c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

OverallQual指的是房子的整体建材质量和品质。在美国建新房是可以真正设计自己的房子的，在design center你可以指定房子内外墙、柜子、五金、窗户、楼梯扶手等部件的材质和颜色。装修过人都知道装潢是很复杂的，单单是门把手，建材商随手就能翻出十几种选择，除此之外，美国的部分家具，如洗碗机、中央空调等它们都属于前装的，也就是在建房的时候安装的，有些老房子就因为房型结构的原因而安装不了中央空调。

因此，OverallQual值越高就代表房子越高档而且往往也越新，价格自然也更贵。新房相比老房也有很大的好处，比如更保暖（墙体内的石棉会老化）、装有中央空调、房型设计更合理等，当然售价也会比老房更高。因此，从点阵图可以看到，OverallQual和SalePrice呈现正相关性，而且它的重要性远远超其他特征。

- **GrLivArea: Above grade (ground) living area square feet**

![](https://upload-images.jianshu.io/upload_images/13575947-10c525c4af27b73d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

GrLivArea是第二重要的特征，它指的是地面以上的居住面积，包括卧室、厨房、客厅等，但不包括地下室和房子外的走廊过道等。它也和售价呈正相关性。

如果你仔细看重要性排名前20的特征就会发现它们大多都是房子的面积，如地下室面积、1楼总面积、2楼总面积、地下室完成部分面积、车库面积等等，既然房价与居住面积呈正相关性，我们就可以围绕居住面积展开更多的维度，详细见下文。

- **CentralAir: Central air conditioning**

![](https://upload-images.jianshu.io/upload_images/13575947-1a18739a1c574b65.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到安装了中央空调的房子比没装的要高，但前面也分析了，有中央空调的房子很可能是装修较好的新房，因此房价高可能是OverallQual导致的。下文我会介绍一种方法，用来分析有无中央空调对整个房价的定量影响。

- **YearBuilt: Original construction date　&　YearRemodAdd: Remodel date**

![](https://upload-images.jianshu.io/upload_images/13575947-80794ab2eca368d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/13575947-550308d2a8a78e72.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

YearBuilt和SalePrice的点阵图比较凌乱，因此这里我用ggplot画它的趋势图。可以看到1940年之后房价是整体上涨的，这是符合美国国情的，二战时期大量的军备订单让美国走出20世纪初的经济大萧条，并在战后成为全球霸主进入黄金发展期，而且随着美元和黄金脱钩之后，通货膨胀成为导致房价上涨的第一因素。

YearRemodAdd是房屋翻修的年份，它很大程度上反映的是OverallQual，从信息论的角度上说，YearBuilt比YearRemodAdd的信息量更大，因此也解释了前者的特征重要性比后者高。

## Feature Engineering

```
df = pd.concat([train_df, test_df], ignore_index=True)
idxs = df[df['GarageYrBlt'].isna()].index
df.loc[idxs, 'GarageYrBlt'] = df.loc[idxs, 'YearBuilt']
df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
df['YBElapsed'] = df['YrSold'] - df['YearBuilt']
df['YRElapsed'] = df['YrSold'] - df['YearRemodAdd']
df.drop('Id', 1, inplace=True)

def bsmtsf2gla(x):
  sf = x.GrLivArea
  if x.BsmtFinType1 != 'LwQ' and x.BsmtFinType1 != 'Unf':
    if x.BsmtFinSF1 != np.nan:
      sf += x.BsmtFinSF1
  if x.BsmtFinType2 != 'LwQ' and x.BsmtFinType2 != 'Unf':
    if x.BsmtFinSF2 != np.nan:
      sf += x.BsmtFinSF2
  return sf

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalLivArea'] = df[['GrLivArea', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']].apply(bsmtsf2gla, axis=1)
df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df.to_csv(PATH/'features.csv', index=False)
!cp {PATH}/'features.csv' drive/house_price/ -v
```

这个项目的特征工程主要围绕房屋面积来展开，增加了几个新特征：
- TotalSF。房屋地上地下总面积。
- TotalLivArea。总的可居住面积。GrLivArea没有包括地下室，但如果是已经建好的、装修良好的地下室是可以用作居住的，而实际情况也是如此，美国家庭地下室常用做娱乐室、洗衣房和卧室。
- TotalBathrooms。美国家庭的人口比较多，卫生间个数也是购房考虑的重要因素。
- TotalPorchSF。汇总所有门廊、走廊、过道面积。

除此之外，我还填充了部分缺失的GarageYrBlt字段，增加了表示房屋年龄的YBElapsed和YRElapsed字段。

## Random Forest

在最终的模型训练之前，需要根据特征重要性（feature importance），筛选掉那些不那么重要的特征（小于0.003）。可以看到，OverallQual和居住面积依旧是相关性最强的两类特征。

![](https://upload-images.jianshu.io/upload_images/13575947-6f3fbd2dddfb562b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
m = fit(min_samples_leaf=1)

[0.983871577939327,
 0.8859291532414196,
 0.8826073971499095,
 0.13614699464846672]
```

初步训练后，模型对验证集（20%训练样本）的RMSE是0.136，如果用完整的数据集来训练，RMSE会降低到0.128~0.13（Final section）。

打开Kaggle的LB可以看到，0.13并不是个好成绩，这主要是因为数据量实在是太少了，训练集只有1500个样本，单个模型的bias会比较大，Kaggle上那些排名较高的模型大都是将多个不同模型ensembling的结果，用系统误差抵消总的bias。Ensembling不是本文的重点，如果你对它感兴趣，可以到[kernels](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/kernels)上找这类模型。

在真实的商业场景中，0.13和0.1并没有什么区别，公司关心的不是具体的预测值，而是怎么用它来盈利。假如你是房产投资人，既然已经知道OverallQual和SalePrice有很强的正相关性，那你最应该做的是找到OverallQual值较低价格合理的房屋买下来，翻修后再卖出去赚个好价钱，但问题来了，你有多相信OverallQual这个指标？为了收益最大化，你要买哪个OverallQual级别的房子？按哪个OverallQual级别来翻修？

## Model Interpretation

2018年，有一位诺贝尔经济学奖得主说到：“人工智能其实就是统计学，只不过用了一个华丽辞藻”。这里不分析这样说法是否片面，的确，机器学习是基于统计学的，但如果把两者划等号，统计学家估计是不答应的。

模型的主要目有两个：预测和解释数据。统计学使用的数据模型（data model）对数据有很好的解释性，但缺点是这种模型的预测能力不强。机器学习的算法模型（algorithm model）有很强的预测能力，但却因为模型过于复杂而无法很好地解释预测结果是怎么来的，因此不被统计学家所认同。

Random Forest是少数具有良好数据解释性的机器学习模型。接下来我会从置信区间和单一特征的定量分析这两方面入手，向你介绍Random Forest是如何解释预测结果的。

### confidence interval

回到之前的问题，模型对OverallQual的预测真的准确吗？误差范围是多大？ 这就需要计算出预测结果的置信区间，RF可以通过决策树的方差来计算置信区间的。

```
preds = np.stack([t.predict(val_x) for t in m.estimators_])
x = val_x.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x['SalePrice'] = val_y
flds = ['OverallQual', 'SalePrice', 'pred', 'pred_std']
oq_summ = x[flds].groupby('OverallQual', as_index=False).mean()
oq_summ

OverallQual	SalePrice	pred	pred_std
0	2	10.471978	11.196668	0.259531
1	3	11.166245	11.333459	0.271934
2	4	11.535102	11.616779	0.173371
3	5	11.773697	11.798080	0.159453
4	6	12.003492	11.997250	0.137785
5	7	12.222000	12.198389	0.139262
6	8	12.477638	12.470133	0.153298
7	9	12.753940	12.727588	0.164472
8	10	13.100623	12.917763	0.159669
```

![不同OverallQual的样本数](https://upload-images.jianshu.io/upload_images/13575947-5b6dc1a792ea6a5f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

preds是所有决策树对同一个数据集（验证集）的预测结果，通过计算preds的标准差就可以得到模型预测的置信度和置信区间，标准差越大表示置信度越低。可以看到，模型对OverallQual 6/7的预测最准确，而对OverallQual 2的预测误差偏大，最主要的原因是OverallQual 2的训练样本太少（只有2个），其实OverallQual 1也存在这个问题，只是因为val_x中没有OverallQual 1的样本而已。

有了均值和标准差，就可以根据得到相应置信度（68-95-99）的置信区间和具体的误差范围。

### PDP（partial dependence plot）

现在我们已经知道了模型对不同OverallQual级别样本的预测准确度，那接下来就要对不同级别的OverallQual与房价之间的影响做定量分析，找出让房价增长最显著的OverallQual区间。

虽然前文的点阵图展示了OverallQual和SalePrice的正相关性，但这很可能是OverallQual和其他特征共同作用下的结果。要排除其他特征的影响，就需要在只改变OverallQual的情况下，观察SalePrice的变化。

假设有10套房子，它们的房型、房屋面积、车库、绿化情况等因素完全相同，唯一的区别就在于OverallQual级别不同（1~10），它们在相同的时间被卖给同一个人，那这10个价格的差值就可以作为OverallQual对房价涨跌幅度的定量影响。

PDP会用1~10填充数据集中的OverallQual字段，这样一来就有了10份只改变了OverallQual的数据集，并用模型分别预测这些数据集，最后将这些预测结果绘制成图形：

![](https://upload-images.jianshu.io/upload_images/13575947-9fb4c29023d7c1d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图中蓝线指的是同一个数据样本的10个不同预测结果，中间黄色的那条线是所有样本的均值，可以看到曲线整体是上升的，也就是说OverallQual的确跟SalePrice是正相关，其中，6和7之间坡度是最陡的，因此可以得出结论，在投入最小风险最低的情况下让收益最大化，应该买入4~6级的房屋装修到7级后卖出。

你还可以用相同方法分析ExterQual、CentralAir、kitchenQual、FireplaceQu等，看看装没装中央空调、不同等级的外墙装修、不同厨房、不同壁炉这些因素对房价的定量影响，将房产投资买卖的收益最大化。

## END

本文以Kaggle "House Prices"项目入手，介绍了这类数据科学项目的基本流程，以及我对机器学习应用到房产投资的思考，最后重点介绍了如何用Random Forest来解释数据。

## Refences

- [【Predict Future Sales】玩转销量预测 part2](https://www.jianshu.com/p/1f6eef8a86fd)
