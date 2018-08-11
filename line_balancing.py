import pandas as pd
import numpy as np
import math


tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
tasks_time = [45, 11, 9, 50, 15, 12, 12, 12, 12, 8, 9]
tasks_parent = [None, 'A', 'B', None, 'D', 'C', 'C', 'E', 'E', ['F', 'G', 'H', 'I'], 'J']
followers = [6, 5, 4, 5, 4, 2, 2, 2, 2, 1, 0]

# column_label = ['Tasks', 'Tasks Time (s)', 'Task that must precede']

dataset = {'Tasks': tasks, 'Tasks Time (s)': tasks_time, 'Tasks that must precede': tasks_parent}

df = pd.DataFrame(data = dataset)


# for i in range(len(tasks)):
# 	followers.append(0)
# print(followers)


print(df)

__idle__time__original = 0

for i in range(0, len(tasks)):
	__idle__time__original += (50 - df.iloc[i]['Tasks Time (s)'])

# __idle__time__original = (50 - df.iloc[0]['Task Time (s)']) + (50 - df.iloc[0]['Task Time (s)'])


import networkx as nx
G = nx.DiGraph()

total_tasks_time = 0

for _, row in df.iterrows():
	# print(row[])
	if row['Tasks that must precede'] is not None:
		if len(row['Tasks that must precede']) > 1:
			for i in row['Tasks that must precede']:
				G.add_edges_from([(i, row['Tasks'])])
		else:
			G.add_edges_from([(row['Tasks that must precede'], row['Tasks'])])


import matplotlib.pyplot as plt



plt.subplot(211)
pos = nx.spring_layout(G, scale = 15)
nx.draw(G, pos, font_size = 8, with_labels=True, font_weight='bold')
plt.savefig('graph.png')
plt.show()

df['Followers'] = followers
print(df)

for _, rows in df.iterrows():
	# print(rows['Followers'])
	total_tasks_time += rows['Tasks Time (s)']
print(total_tasks_time)

Production_time_per_day = 420 * 60
Wagon_required = 500

CT = Production_time_per_day / Wagon_required
minimum_workstation = math.ceil(total_tasks_time / CT)

print(CT, minimum_workstation)

df_copy = df.copy()
df_copy.sort_values(by = ['Followers', 'Tasks Time (s)'], inplace = True, ascending = False)

print(df_copy)

l = []

cnt = 0

for _, rows in df_copy.iterrows():
	l.append(tuple((rows['Tasks'], rows['Tasks Time (s)'])))

print(l)

a = []
yy = 0
for i in range(0, len(tasks)):
	# a[i] = l[i]
	if i is not yy:
		continue
	# a.append(tuple((i, l[i])))
	idle_time = CT - l[i][1]
	a.append(tuple((i, l[i], idle_time)))
	# print(idle_time)

	for j in range(i+1, len(tasks)):
		if l[j][1] <= idle_time:
			# print("aaaa ", idle_time)
			# a.append(tuple((i, l[j])))
			idle_time -= l[j][1]
			a.append(tuple((i, l[j], idle_time)))
		else:
			yy = j
			i = j
			break

print(a)

column__label = ['Order', 'Tasks Process_Time', 'Idle Time (s)']
df = pd.DataFrame(a, columns = column__label)
print(df)

__idle__time = df.iloc[0]['Idle Time (s)'] + df.iloc[1]['Idle Time (s)'] + df.iloc[5]['Idle Time (s)'] + df.iloc[9]['Idle Time (s)'] + df.iloc[10]['Idle Time (s)']

print(__idle__time__original)
print(__idle__time)


# __idle__time = df.iloc[0].values + df.iloc[1] + df.iloc[5] + df.iloc[9] + df.iloc[10]
# print(__idle__time)

# dat = dict(a)

# print(dat)

# d = {}
# for k, v in a:
#     d[k] = d.get(k, ()) + (v,)
# print (d)

# workstation = pd.DataFrame(data = d)

# print(workstation)