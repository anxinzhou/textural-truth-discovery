import os
import shutil


# base_dir = "answer_grading/data/scores"
# dst_dir = "answer_grading/scores"
# a= os.listdir(base_dir);
# a = [eval(e) for e in a ]
# a.sort()
# a=[str(e) for e in a]
# print(a)
# for i, dir in  enumerate(a):
#     src_path = os.path.join(base_dir, dir, "ave")
#     dst_path = os.path.join(dst_dir, str(i))
#     shutil.copyfile(src_path,dst_path )


# base_dir =  "answer_grading/answer"
# os.listdir(base_dir)
# a= os.listdir(base_dir);
# a.sort(key=lambda x: len(x))
# a.sort(key=lambda x: eval(x))
# print(a)
#
# for i, file in  enumerate(a):
#     src_path = os.path.join(base_dir, file)
#     dst_path = os.path.join("answer_grading/tmp2_answers", str(i))
#     shutil.move(src_path,dst_path )

# base_dir = "answer_grading/questions"
# dst_dir = "answer_grading/questions"
# a=os.listdir(base_dir)
# for file in a:
#     f = open(os.path.join(base_dir,file),'rb')
#     content = f.readlines()
#     to_write = []
#     for c in content:
#         c=c.split()
#         c=c[1:]
#
#         c= b' '.join(c)
#         c+=b'\n'
#         to_write.append(c)
#     f2 = open(os.path.join(dst_dir,file),'wb')
#     f2.writelines(to_write)
#     f.close()
#     f2.close()

f = open("glove.6B/glove.6B.50d.txt")
content = f.readlines()
for c in content:
    c = c.strip()
    print(c)
    break