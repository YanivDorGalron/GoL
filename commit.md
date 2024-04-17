# How to Write Better Git Commit Messages

Writing clear and informative Git commit messages is crucial for effective collaboration and code maintenance. Follow these steps to improve your commit messages:

1. **Capitalization and Punctuation**: Capitalize the first word and avoid ending with punctuation. Use all lowercase for Conventional Commits.
2. **Mood**: Use imperative mood in the subject line for a concise and actionable message.
3. **Type of Commit**: Specify the type (e.g., feat, fix, chore) to categorize your changes. You can use Conventional Commits for a structured format:
   - **feat**: A new feature is introduced with the changes.
   - **fix**: A bug fix has occurred.
   - **chore**: Changes that do not relate to a fix or feature and don't modify source or test files (e.g., updating dependencies).
   - **refactor**: Refactored code that neither fixes a bug nor adds a feature.
   - **docs**: Updates to documentation such as the README or other markdown files.
   - **style**: Changes that do not affect the meaning of the code, likely related to code formatting (e.g., white-space, missing semi-colons).
   - **test**: Includes new or correcting previous tests.
   - **perf**: Performance improvements.
   - **ci**: Continuous integration related.
   - **build**: Changes that affect the build system or external dependencies.
   - **revert**: Reverts a previous commit.

Example of a Conventional Commit:

feat: improve performance with lazy load implementation for images

This commit fixes the broken behavior of the component by implementing lazy loading for images.

By following these guidelines and thinking like a journalist, you can create commit messages that are clear, informative, and helpful for future developers.

Check out [freeCodeCamp's detailed guide](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/) for more insights.