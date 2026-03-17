# Contributing Guide

This guide explains how to contribute to the project by **forking the repository, setting up the project in VS Code, and creating a separate branch for your work**.

---

# 1. Fork the Repository

1. Open the project repository on GitHub.
2. Click the **Fork** button at the top-right.
3. This creates a copy of the repository in your GitHub account.

Example:

```
Original Repository → github.com/project-owner/project-name
Your Fork           → github.com/your-username/project-name
```

---

# 2. Clone the Forked Repository

Open **VS Code Terminal**, **PowerShell**, or **Git Bash** and run:

```
git clone https://github.com/your-username/project-name.git
```

Move into the project folder:

```
cd project-name
```

---

# 3. Open the Project in VS Code

From inside the project directory run:

```
code .
```

This opens the project folder directly in **VS Code**.

---

# 4. Create a Virtual Environment

It is recommended to create a virtual environment before installing dependencies.

```
python -m venv venv
```

Activate it.

**Windows**

```
venv\Scripts\activate
```

**Linux / macOS**

```
source venv/bin/activate
```

---

# 5. Install Project Dependencies

Install all required libraries using the requirements file.

```
pip install -r requirements.txt
```

This installs all packages needed to run the project.

---

# 6. Check Current Branch

Before creating a new branch, check which branch you are on.

```
git branch
```

Usually the default branch will be:

```
main
```

---

# 7. Create a New Branch

Always create a **separate branch** before starting any work.

```
git checkout -b branch-name
```

Example:

```
git checkout -b preprocessing-update
```

After running this command, you will switch to the new branch.

---

# 8. Verify the New Branch

You can confirm that the new branch is active by running:

```
git branch
```

The active branch will appear with a `*` symbol.

Example:

```
main
* preprocessing-update
```

---

# Workflow Summary

```
Fork Repository
       ↓
Clone Repository
       ↓
Open Project in VS Code
       ↓
Create Virtual Environment
       ↓
Install Dependencies
       ↓
Create a New Branch
       ↓
Start Working on the Project
```

---
