'''
process data
author:wangbei
2018/12/22 20:26
'''
import os
datapath = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework4/data/"
infile = os.path.join(datapath,"dataset.txt")  # conference:BigMine appears 69 times,Enabling appears 47 times
outfile = os.path.join(datapath,"DBLP.txt")
conference_list = ['IJCAI', 'AAAI', 'COLT', 'CVPR', 'NIPS', 'KR', "SIGIR",'KDD']
output = open(outfile, 'w')
with open(infile,'r') as infile:
    while True:
        line = infile.readline()
        if not line:
            break
        author_list, title, year, conference = [], '', '', ''
        while not line.startswith("#######") and line:
            if line.startswith("author"):
                author_list.append(line.strip().split("\t")[1])
            elif line.startswith("title"):
                title = line
            elif line.startswith("year"):
                year = line
            else:
                conference = line.strip().split("\t")[1]
            line = infile.readline()
        for conf in conference_list:
            if conf.lower() in conference.lower():
                conference = conf
                output.write("author" + '\t' + '\t'.join(author_list) + '\n')
                output.write(title if title != '' else 'title' + '\t' + '\n')
                output.write(year if year != '' else 'year' + '\t' + '\n')
                output.write("conference" + '\t' + conference + '\n' if conference != '' else 'conference' + '\t' + '\n')
                output.write("\n")
output.close()