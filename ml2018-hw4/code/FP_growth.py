# -*- coding: utf-8 -*-
'''
task 1: To find the supporters of the conferences
author:wangbei
2018/12/23
'''
import os
datapath = "/media/bei/94AA9020AA8FFD44/MachineLearning2018/homework4/data/"
infile = os.path.join(datapath,"DBLP.txt")
outfile1 = os.path.join(datapath,"researches_group.txt")

def read_data(infile):
    res, res_dict, author_papers, paper_year = [], {}, {}, {}
    with open(infile,'r') as f:
        while True:
            author_list = f.readline().strip().split("\t")[1:]
            # print "@author_list",author_list
            title = f.readline().strip().split("\t")
            if len(title)>1:
                title = title[1]
            else:
                title = "" 
            # print title
            year = f.readline().strip().split("\t")
            if len(year)>1:
                year = int(year[1])
            else:
                year = ""

            conference = f.readline().strip().split("\t")
            if len(conference) >1:
                conference = conference[1]
            else:
                conference = ""
            for author in author_list:
                if res_dict.has_key(author):
                    res_dict[author] = res_dict[author] + 1  # res_dict = {author1:paper_num,...,authorN:paper_num}
                else:
                    res_dict[author] = 1
            res.append(author_list)  # res = [[paper1_author_list],[paper2_author_list],...,[paperN_author_list]]
            for author in author_list:
                if author_papers.has_key(author):    #author_papers = {author1:[title1,...titleN],...}
                    author_papers[author].append(title)
                else:
                    author_papers[author] = [title]
            paper_year[title] = year    # paper_year = {title1:year,....titleN:yeaer}
            line = f.readline()
            # print "#######",line
            if len(line) == 0:
                break
            if not line:
                break

    # print "len(res): ",len(res),res[-1]
    # del res[-1]
    # print "len(res): ",len(res),res[-1]
    return res, res_dict, author_papers, paper_year

def sort_authors(author_lists, author_paper_num):
    #sort the authorlist: author_paper_num:from large to small; author_name:from small to large
    sort_author_lists = []
    for item in author_lists:
        item = [v for v in item if author_paper_num.has_key(v)]
        item = sorted(item, key=lambda x:(-author_paper_num[x], x), reverse=False)  
        sort_author_lists.append(item)
    return sort_author_lists

# construct conditional fp-tree recursively,to obtain frequent items
def get_frequent_items(base_author, sorted_authors, support):
    '''
        base_author: 频繁项基，初始化为空
        sorted_authors:包含该频繁项基的有序的作者列表
        support:支持度
    '''
    res = [base_author]
    author_num = {}
    #统计和base_author一起发表论文的其他作者发表的论文数量
    for author_list in sorted_authors:
        for author in author_list:
            if author_num.has_key(author):
                author_num[author] += 1 
            else:
                author_num[author] = 1
    # print "author_num: ",author_num
    #过滤掉不符合support阈值的作者
    filtered_authors = [k for k, v in author_num.items() if v >= support]
    # print filtered_authors
    #递归调用该函数，条件基中元素不断扩展，返回的频繁项集也越来越复杂
    for author in filtered_authors:
        res = res + get_frequent_items(base_author + [author], \
                    [author_list[:author_list.index(author)] for author_list in sorted_authors \
                    if author in author_list], support)
        # print res
    return res

def get_team_papers(author_topics, frequent_items_support_3):
    frequent_items_papers = []
    for team in frequent_items_support_3:
        author_papers = set(author_topics[team[0]])
        for team_member in team[1:]:
            # 将团队中每个研究者的论文取交集，得到该团队共同发表的论文
            author_papers.intersection_update(set(author_topics[team_member]))
        frequent_items_papers.append(author_papers)
    teams, team_papers = [], []
    mask = [1] * len(frequent_items_support_3)
    # filter the team which containing more than 4 authors, but it is still the child set of other set
    for index1, team1 in enumerate(frequent_items_support_3):        
        for index2, team2 in enumerate(frequent_items_support_3): 
            if index1 != index2 and mask[index1] == 1 \
            and len(set(team1) & set(team2) ^ set(team1)) == 0 \
            and frequent_items_papers[index1] == frequent_items_papers[index2]:
                mask[index1] = 0
                break
    for index1, team1 in enumerate(frequent_items_support_3):
        if mask[index1]:
            res.append(team1), res_topics.append(frequent_items_papers[index1])
    return teams, team_papers

if __name__ == '__main__':
    support = 3
    author_lists, author_paper_num, author_papers, paper_year = read_data(infile)
    sorted_author_lists = sort_authors(author_lists, author_paper_num)
    # print "len(author_lists)",len(author_lists)
    # print "len(sorted_author_lists)",len(sorted_author_lists)
    frequent_items = get_frequent_items([], sorted_author_lists, support) 
    '''
    get all frequent items. e.g.['Ning Chen'], ['Ning Chen', 'Eric P. Xing'], ['Ning Chen', 'Jun Zhu 0001'], 
    ['Ning Chen', 'Jun Zhu 0001', 'Eric P. Xing'], ['Ning Chen', 'Bo Zhang 0010'],
     ['Ning Chen', 'Bo Zhang 0010', 'Jun Zhu 0001'],
    '''
    frequent_items_support_3 = [frequent_item for frequent_item in frequent_items if len(frequent_item) > 3] # filter the frequent items that containing at least four authors
    # print  len(frequent_items_support_3) # 525
     teams, team_papers = get_team_papers(author_papers, frequent_items_support_3)

    with open(outfile1, 'w') as output:
        for index, team in enumerate(teams):
            output.write("\t".join(team) + '\n')
            for paper in sorted(team_papers[index], key=lambda x: paper_year[x]):
                output.write("\t".join(['', paper, str(paper_year[paper])]) + '\n')
            output.write('\n')