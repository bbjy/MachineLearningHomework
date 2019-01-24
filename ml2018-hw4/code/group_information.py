import os
path = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework4/data/"
infile = path + "researches_group.txt"
outfile = path + "group_years.txt"

maxyear = 0
minyear = 2019
teams_year = {}
with open(infile) as f:
	while True:
		line = f.readline()
		if not line:
			break
		while not line == "\n":
			if not line.startswith("\t"):
				team = line		
			elif line.startswith("\t"):
				year = int(line.strip().split("\t")[1])
				if year < minyear:
					minyear = year
				if year > maxyear:
					maxyear = year
			line = f.readline()
		teams_year[team] = maxyear - minyear
		maxyear = 0
		minyear = 2019
sorted_teams_year = sorted(teams_year.items(), key=lambda e: int(e[1]),reverse = True)
# print sorted_teams_year[0],type(sorted_teams_year[0])
of = open(outfile,'w')
for v in sorted_teams_year:
    key = v[0].strip()
    value = v[1]
    of.write(key +"\t" + str(value)+"\n")
of.close()


    





