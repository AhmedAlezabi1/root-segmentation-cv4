# Archived workflows

These GitHub Actions workflows ran the project's CI/CD in the original team repository:

| Workflow | What it did |
|---|---|
| `docker-publish.yml` | Built the backend and frontend Docker images and pushed them to GitHub Container Registry |
| `deploy.yml` | Deployed the published images to Azure Container Apps |
| `drift_check.yml` | Daily data-drift check against the production database |
| `fix-aca-inference-mode.yml` | Manual fix-up job for the Azure deployment |

They are kept for reference only. GitHub only runs workflows in `.github/workflows/`, so these are inactive in this copy, because the cloud resources and secrets they need belong to the original project.
