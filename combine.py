import os

with open("article_methods.qmd", "r") as f:
    methods = f.read().split("---")[-1].strip()

with open("results.qmd", "r") as f:
    results = f.read().split("---")[-1].strip()

# Create combined content
frontmatter = """---
title: "Colorado Pollen Detection: Methods and Results"
format:
  pdf:
    toc: true
    number-sections: false
    colorlinks: true
  docx:
    toc: true
    number-sections: false
---

"""

# Let's adjust results.qmd headers so they fit nicely
# actually they are `## ` so they fit nicely after `# 3. Results`
results_content = "# 3. Results\n\n" + results

combined = frontmatter + methods + "\n\n" + results_content

with open("report.qmd", "w") as f:
    f.write(combined)
print("Created report.qmd")
