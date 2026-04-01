# GitHub API — Answer

---

## 1. What is the role of query parameters in this request?

Query parameters are key-value pairs added to the URL after a `?` symbol.
They allow us to **customize and filter** the API request without changing the endpoint itself.

### In this request we used:

| Parameter  | Value    | Purpose                              |
|------------|----------|--------------------------------------|
| `q`        | `python` | Search keyword — what to look for    |
| `sort`     | `stars`  | Sort results by number of stars      |
| `order`    | `desc`   | Show highest stars first             |
| `per_page` | `5`      | Limit the results to only 5 repos    |

### Example URL with query parameters:
```
https://api.github.com/search/repositories?q=python&sort=stars&order=desc&per_page=5
```

Without query parameters, the API would return a default, unfiltered list —
we would have no control over what results we get back.

---

## 2. Why do we use `response.json()` instead of `response.text`?

| Feature         | `response.text`                        | `response.json()`                        |
|-----------------|----------------------------------------|------------------------------------------|
| Return type     | Plain string                           | Python dictionary / list                 |
| Usability       | Hard to work with directly             | Easy to access with keys like `data["items"]` |
| Parsing needed? | Yes — you'd need `json.loads()` manually | No — already parsed automatically       |
| Best for        | Debugging raw responses                | Extracting and using structured data     |

### Example:

```python
# response.text returns a raw string like this:
'{"total_count": 1234, "items": [{"full_name": "python/cpython", ...}]}'

# response.json() returns a Python dictionary — easy to use:
data = response.json()
print(data["items"][0]["full_name"])  # Output: python/cpython
```

### Summary:
We use `response.json()` because the GitHub API returns data in **JSON format**,
and `response.json()` automatically converts it into a **Python dictionary**,
making it easy to extract specific fields like `full_name` and `stargazers_count`.