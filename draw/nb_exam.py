import numpy as np
import matplotlib.pyplot as plt

oblivious_file_name = 'oblivious.txt'
non_oblivious_file_name = 'non-oblivious.txt'

def read_file(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        lines = [eval(line.strip()) for line in lines]
    return np.array(lines).reshape(-1,10)

non_oblivious_exam_score = read_file(non_oblivious_file_name)
oblivious_exam_score = read_file(oblivious_file_name)

for i, (non_oblivious_score, oblivious_score) in enumerate(zip(non_oblivious_exam_score, oblivious_exam_score)):
    exam_id = i + 1
    plt.clf()
    plt.ylim(3.5, 5)
    plt.plot(range(1, 11), non_oblivious_score, marker='o', label='non oblivious')
    plt.plot(range(1, 11), oblivious_score, marker='x', label='oblivious')
    plt.xlabel("Top-K")
    plt.ylabel("Average Score")
    plt.legend()
    fname = "Exam " + str(exam_id)
    plt.savefig(fname)
