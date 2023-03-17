import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris_data.csv")

fig,ax=plt.subplots(1,1,figsize=(21,10))
setosa=data[data["species"]=="Iris-setosa"]
plt.scatter(data.sepal_width,data.sepal_length)
setosa.plot(x="sepal_length",y="sepal_width",kind="hist",ax=ax,label='setosa',color='red')
ax.set(title='sepalcomparision',ylabel="sepal_width")
ax.legend()
plt.show()