import numpy as np
import matplotlib.pyplot as plt

exam_score = np.array(
    [4.21429, 4.28571, 4.30952, 4.48214, 4.28571, 4.21429, 4.20408, 4.16964, 4.07937, 4.05, 3.85714, 4,
     3.85714, 3.96429, 4.02857, 4.07143, 4.09184, 4.11607, 4.12698, 4.15714, 4.21429, 4.28571, 4.40476,
     4.53571, 4.62857, 4.61905, 4.55102, 4.49107, 4.51587, 4.5, 4.5, 4.46429, 4.42857, 4.375, 4.45714, 4.40476,
     4.37755, 4.28571, 4.30159, 4.27857, 4.125, 4.3125, 4.29167, 4.15625, 4.2, 4.14583, 4.16071, 4.23438,
     4.29167, 4.3125, 4.71429, 4.60714, 4.57143, 4.58929, 4.51429, 4.45238, 4.43878, 4.46429, 4.44444, 4.49286,
     4.14286, 4.32143, 4.28571, 4.28571, 4.35714, 4.38095, 4.42857, 4.40179, 4.34921, 4.32857, 4.57143,
     4.35714, 4.5, 4.44643, 4.47143, 4.53571, 4.54082, 4.50893, 4.49206, 4.48571, 4.42857, 4.53571, 4.66667,
     4.73214, 4.62857, 4.60714, 4.55102, 4.57143, 4.57143, 4.55714, 4.57143, 4.67857, 4.45238, 4.53571,
     4.41429, 4.42857, 4.42857, 4.4375, 4.4127, 4.45714, 4.65, 4.3, 4.38333, 4.425, 4.32, 4.30833, 4.23571,
     4.16875, 4.16111, 4.17, 4.4, 4.4, 4.53333, 4.5875, 4.56, 4.47708, 4.42321, 4.47344, 4.47639, 4.51875])
exam_score = exam_score.reshape(12,-1)
print(exam_score)
for i,score in enumerate(exam_score):
    exam_id = i+1
    plt.clf()
    plt.ylim(3.5,5)
    plt.plot(range(1,11),score,marker='o')
    plt.xlabel("Top-K")
    plt.ylabel("Average Score")
    fname = "Exam "+str(exam_id)
    plt.savefig(fname)