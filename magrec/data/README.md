# `data` folder

Here all datasets are contained. Their names, descriptions, and files should be in separate subfolders.
That's where not public names and results can be stored. Separately, each subfolder with data can be
versioned with git, which is recommended. All subfolders are ignored by magrec git by default.

in .gitignore
```
data/*/*  # ignore all files in data subfolders
```