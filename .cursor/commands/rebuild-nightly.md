# Rebuild Nightly Docker Image

Pull the latest changes from upstream into main, ensure you are on main, sync submodules, and rebuild the Docker image.

## Steps

1. **Save current branch**: Run `git branch --show-current` to note the current branch name.

2. **Update main from upstream**:

   ```bash
   git switch main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

3. **Go back to main**: Ensure you are on `main` and up to date:

   ```bash
   git switch main
   git pull origin main
   ```

4. **Sync submodules**:

   ```bash
   git submodule update --init --recursive
   ```

5. **Build Docker image**:

   ```bash
   docker buildx build --build-context nemo-rl=. -f docker/Dockerfile --tag nemo-rl:latest-cache .
   ```
   Make sure to use nohup with a corresponding .log file to track the build.

6. **Report result**: Wait for the Docker build to complete and report whether it succeeded or failed.
