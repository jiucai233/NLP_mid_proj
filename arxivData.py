import arxiv
import pandas as pd

# 创建搜索查询
search = arxiv.Search(
  query = "cs.AI",
  max_results = 10,  # 限制结果数量
  sort_by = arxiv.SortCriterion.LastUpdatedDate
)

# 创建一个列表来存储结果
data = []

# 循环遍历结果
for result in search.results():
    # 获取作者姓名
    authors = ", ".join([author.name for author in result.authors])

    # 将数据添加到列表中
    data.append([result.title, authors, result.summary.replace('\n', ' '), str(result.published)])

# 创建 Pandas DataFrame
df = pd.DataFrame(data, columns=['Title', 'Authors', 'Abstract', 'Date'])
df.to_pickle("data/arxiv_papers.pkl")  # 保留完整 Python 对象结构
