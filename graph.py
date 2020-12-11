# from matplotlib import pyplot as plt
# X=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# Y=[0.82132703,0.8175355,0.82180095,0.8004739,0.8165877,0.7421801,0.81327015,0.6620853,0.8260664]
# plt.plot(X, Y)
# # plt.legend(["train_acc", "test_acc"])
# plt.xlabel(r"Threshold $\tau$")
# plt.ylabel("Micro F1-score")
# plt.savefig('./attention/savedmodel/choose_t.jpg')
# plt.title('AG-CNN')
# plt.show()



import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['VGG16','VGG19','InceptionV3','MobileNet','ResNet50','AG-CNN']
men_means = [0.89,0.91,0.90,0.90,0.900,0.850]
women_means = [0.900,0.890,0.890,0.890,0.900,0.840]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Accuracy')
rects2 = ax.bar(x + width/2, women_means, width, label='Micro F1-score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Model Name')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim((0.7,1))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('./savedmodel/choose_t.jpg')
plt.show()