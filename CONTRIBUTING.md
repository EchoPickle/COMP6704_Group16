# Contributing

Repository owner: invite the other members as collaborators under **Settings → Collaborators → Add people**. Each member must accept the invitation before pushing to this repository.

Keep each algorithm in its own top-level directory, named after the algorithm. Include runnable code, dependencies, a README explaining the objective/data/preprocessing/commands, and any results or figures needed for the project. Avoid changing another member's files without discussion.

## Git workflow

Clone the repository and create a branch for your implementation:

```bash
git clone https://github.com/EchoPickle/COMP6704_Group16.git
cd COMP6704_Group16
git switch -c add-your-algorithm
```

Copy your files into your algorithm directory. Replace `YOUR_ALGORITHM` below with that directory's actual name:

```bash
git add YOUR_ALGORITHM/
git commit -m "Add algorithm implementation"
git push -u origin add-your-algorithm
```

On GitHub, open a pull request from your branch to `main`; another group member can review it before merging. Before later work, switch to `main` and run `git pull`, then create a new branch.

## Browser upload

On the repository page, choose **Add file → Upload files**, drag in your algorithm folder, write a commit message, and choose a new branch. Open a pull request to `main`. GitHub browser uploads allow up to 100 files at once and 25 MiB per file; use Git or GitHub Desktop for larger batches.

Do not commit passwords, API keys, virtual environments, caches, raw files containing personal identifiers, or unrelated coursework documents. Keep shared train/validation/test splits and the SVM objective aligned when making cross-algorithm comparisons; otherwise clearly document differences.
