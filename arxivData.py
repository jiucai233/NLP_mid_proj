import arxiv
import pandas as pd

# Create search query
search = arxiv.Search(
  query = "cs.AI",
  max_results = 100,  # Limit the number of results
  sort_by = arxiv.SortCriterion.LastUpdatedDate
)

# Create a list to store the results
data = []

# Loop through the results
for result in search.results():
    # Get author name
    authors = ", ".join([author.name for author in result.authors])

    # Add the data to the list
    data.append([result.title, authors, result.summary.replace('\n', ' '), str(result.published)])

# Create Pandas DataFrame
df = pd.DataFrame(data, columns=['Title', 'Authors', 'Abstract', 'Date'])
df.to_pickle("data/arxiv_papers.pkl")  # Retain the complete Python object structure
