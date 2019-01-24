'''
task 1: To find the supporters of the conferences
author:wangbei
2018/12/22 
'''
import os
datapath = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework4/data/"
infile = os.path.join(datapath,"DBLP.txt")
outfile1 = os.path.join(datapath,"conference_researchers.txt")
outfile2 = os.path.join(datapath,"conference_year_researchers.txt")
outfile3 = os.path.join(datapath,"conference_researchers_activate.txt")

conference_supporter = {'IJCAI': {}, 'AAAI': {}, 'COLT': {}, 'CVPR': {}, 'NIPS': {}, 'KR': {}, 'SIGIR': {}, 'KDD': {}}
conference_year_supporter = {'IJCAI': {}, 'AAAI': {}, 'COLT': {}, 'CVPR': {}, 'NIPS': {}, 'KR': {}, 'SIGIR': {}, 'KDD': {}}

with open(infile,'r') as f:
    while True:
        author_list, title, year, conference = [], '', '', ''
        line = f.readline()
        if line=='':
            break
        author_list = line.strip().split("\t")[1:]
        title = f.readline().strip().split('\t')[1]
        year = int(f.readline().strip().split('\t')[1])
        conference = f.readline().strip().split('\t')[1]
        f.readline()
        for author in author_list:
            if conference_supporter[conference].has_key(author):
                conference_supporter[conference][author].append(year)
            else:
                conference_supporter[conference][author] = [year]
        if conference_year_supporter[conference].has_key(year):
            conference_year_supporter[conference][year].extend(author_list)
        else:
            conference_year_supporter[conference][year] = author_list

with open(outfile1, 'w') as output:
    for conf, value in conference_supporter.items():
        output.write(conf+"\n")
        for auth, year_list in value.items():
            output.write(auth + "\t" + "\t".join([str(year) for year in sorted(year_list)])+"\n")
        output.write("\n")

with open(outfile2, 'w') as output:
    for conf, year_supporter in conference_year_supporter.items():
        output.write(conf+"\n")
        for year, supporter_list in sorted(year_supporter.items()):
            output.write(str(year) + "\t" + "\t".join(set(supporter_list)) + "\n")
        output.write("\n")

activate_supports = []
with open(outfile3,'w') as output:
	for conf,year_supporter in conference_year_supporter.items():
		output.write(conf+"\n")
		for year,supporter_list in sorted(year_supporter.items()):
			if len(activate_supports)==0:
				activate_supports = supporter_list
			else:
				activate_supports = list(set(activate_supports).intersection(set(supporter_list)))
		output.write("\t".join(activate_supports)+"\n")
			


