import os
import matplotlib.pyplot as plt
import numpy as np
y = [0,55]
x = [0,55]
# dataset : iris,bank,letter,car 
MH_train10 = [1.67,0.55,3.85,7.85]  # error percent %
MR_train10 = [3.33,3.65,50,19.97]

MH_train100 = [0,0,3.79,2.46]  #
MR_train100 = [0,0,29.07,3.33]

MH_train1000 = [0,0,2.83,1.23]  #
MR_train1000 = [0,0,22.83,1.16]

MH_test10 = [1.11,0.73,3.85,9.25]  #
MR_test10 = [3.33,6.18,50,24.28]

MH_test100 = [1.11,0.36,3.80,3.47]  #
MR_test100 = [3.33,1.09,29.35,6.36]

MH_test1000 = [1.11,0.36,2.84,2.60]  #
MR_test1000 = [1.11,0.36,2.84,2.60]

plt.figure(1)
plt.title("Train error after 10 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_train10,MH_train10,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Train10.png")

plt.figure(2)
plt.title("Train error after 100 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_train100,MH_train100,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Train100.png")

plt.figure(3)
plt.title("Train error after 1000 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_train1000,MH_train1000,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Train1000.png")

plt.figure(4)
plt.title("Test error after 10 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_test10,MH_test10,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Test10.png")

plt.figure(5)
plt.title("Test error after 100 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_test100,MH_test100,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Test100.png")

plt.figure(6)
plt.title("Test error after 1000 rounds")
plt.xlim(0,50)
plt.ylim(0,50)
plt.plot(x,y,ls=':')
plt.scatter(MR_test1000,MH_test1000,marker='s',s=30)
plt.xlabel("AdaBoost.MR")
plt.ylabel("AdaBoost.MH")
plt.savefig("resultimage/Test1000.png")

